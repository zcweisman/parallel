#include <mpi.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include "/home/ubuntu/Cluster/AntTweakBar/include/AntTweakBar.h"
#include "utils.h"
#include "cutil_math.h"

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600

using namespace std;

typedef struct __attribute__((packed)) {
	uint8_t type;
	float3 cam_pos;
	float cam_rot;
	float3 a, b, c;
	float3 light_pos;
	float3 light_col;
	float delta_t;
	float3 box_min;
	float3 box_max;
	uint32_t frame_seq;
} Request;

typedef struct {
	uint32_t frame_seq;
} FrameHeader;

typedef struct {
	unsigned int* host_mem;
	unsigned int* dev_mem;
} FrameData;

// Globals ---------------------------------------
unsigned int image_width   = IMAGE_WIDTH;
unsigned int image_height  = IMAGE_HEIGHT;

typedef struct {
	uint32_t exp_frame_seq;
	uint32_t frame_seq;
	bool recv_buf[3];
	int buf_size[3];
	unsigned int frame_buf[3*IMAGE_WIDTH*IMAGE_HEIGHT];
} Animation;

GLuint pbo;               // this pbo is used to connect CUDA and openGL
GLuint result_texture;    // the ray-tracing result is copied to this openGL texture
TriangleMesh mesh;

TriangleMesh ground;
TriangleMesh sphere;
TriangleMesh object;
int total_number_of_triangles = 0;

//float *dev_triangle_p; // the cuda device pointer that points to the uploaded triangles

// MPI Globals ----------------------------------
int world_size;
int world_rank;
Request frame_req;
Animation animation;
FrameData frame;
MPI_Status status;

// Camera parameters -----------------------------
float3 a; float3 b; float3 c;
float3 campos;
float camera_rotation = 0;
float camera_distance = 75;
float camera_height = 25;
bool animate = true;

// Scene bounding box ----------------------------
float3 scene_aabbox_min;
float3 scene_aabbox_max;

float light_x = -23;
float light_y = 25;
float light_z = 3;
float light_color[3] = {1,1,1};

// mouse controls --------------------------------
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
bool left_down  = false;
bool right_down = false;

// Main functions
bool initGL();
bool initCUDA();
void initCUDAmemory();
//void Terminate(void);
//void initTweakMenus();
void display();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void KeyboardUpCallback(unsigned char key, int x, int y);
void SpecialKey(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void rayTrace();

bool initGL() {
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "ERROR: GL Failed to initialize on node %d", world_rank);
		fflush(stderr);
		//return CUTFalse;
		return false;
	}
	if (! glewIsSupported
		(
		"GL_VERSION_4_4 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		)) {
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			//return CUTFalse;
            return false;
	}

	// init openGL state
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// view-port
	glViewport(0, 0, image_width, image_height);

	//initTweakMenus();
	return true;
}

bool initCUDA() {
	if (world_rank == 0)
		cudaGLSetGLDevice(0);

	cudaSetDevice(0);
	
	return true;
}

void initCUDAmemory(float* dev_triangle_p) {
	// initialize the PBO for transferring data from CUDA to openGL
	unsigned int num_texels = image_width * image_height;
	unsigned int size_tex_data = sizeof(GLubyte) * num_texels * 4;
	void *data = malloc(size_tex_data);
	// create buffer object
	if (world_rank == 0) {
		glGenBuffers(1, &pbo);
		glBindBuffer(GL_ARRAY_BUFFER, pbo);
		glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
		free(data);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// register this buffer object with CUDA
		cudaGLRegisterBufferObject(pbo);
		//CUT_CHECK_ERROR_GL();
		// create the texture that we use to visualize the ray-tracing result
		glGenTextures(1, &result_texture);
		glBindTexture(GL_TEXTURE_2D, result_texture);

		// set basic parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// buffer data
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}

	// next we load a simple obj file and upload the triangles to an 1D texture.
	loadObj("data/cube.obj",mesh,1);
	loadObj("data/sphere.obj",sphere,1);

	vector<float4> triangles;

	for (unsigned int i = 0; i < mesh.faces.size(); i++) {
		float3 v0 = mesh.verts[mesh.faces[i].v[0]-1];
		float3 v1 = mesh.verts[mesh.faces[i].v[1]-1];
		float3 v2 = mesh.verts[mesh.faces[i].v[2]-1];
		v0.y -= 10.0;
		v1.y -= 10.0;
		v2.y -= 10.0;
		triangles.push_back(make_float4(v0.x,v0.y,v0.z,0));
		triangles.push_back(make_float4(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z,0)); // notice we store the edges instead of vertex points, to save some calculations in the
		triangles.push_back(make_float4(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z,0)); // ray triangle intersection test.
	}

	for (unsigned int i = 0; i < sphere.faces.size(); i++) {
		float3 v0 = sphere.verts[sphere.faces[i].v[0]-1];
		float3 v1 = sphere.verts[sphere.faces[i].v[1]-1];
		float3 v2 = sphere.verts[sphere.faces[i].v[2]-1];
		triangles.push_back(make_float4(v0.x,v0.y,v0.z,0));
		triangles.push_back(make_float4(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z,1)); // notice we store the edges instead of vertex points, to save some calculations in the
		triangles.push_back(make_float4(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z,0)); // ray triangle intersection test.
	}

	cout << "Node " << world_rank;
	cout << " total number of triangles check:" << mesh.faces.size() + sphere.faces.size() << " == " << triangles.size()/3 << endl;

	size_t triangle_size = triangles.size() * sizeof(float4);
	total_number_of_triangles = triangles.size()/3;
	
	int arraySize = image_width*image_height*sizeof(unsigned int);
	
	if(triangle_size > 0 && world_rank != 0) {
		cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaHostAlloc((void**)&frame.host_mem, arraySize, cudaHostAllocMapped);
		cudaHostGetDevicePointer((void**)&frame.dev_mem, (void*)frame.host_mem, 0);
	

		cudaMalloc((void **)&dev_triangle_p, triangle_size);
	
		cudaMemcpy(dev_triangle_p,&triangles[0],triangle_size,cudaMemcpyHostToDevice);
		bindTriangles(dev_triangle_p, total_number_of_triangles);
	}

	frame_req.box_min = mesh.bounding_box[0];
	frame_req.box_max = mesh.bounding_box[1];

	frame_req.box_min.x = min(frame_req.box_min.x,sphere.bounding_box[0].x);
	frame_req.box_min.y = min(frame_req.box_min.y,sphere.bounding_box[0].y);
	frame_req.box_min.z = min(frame_req.box_min.z,sphere.bounding_box[0].z);

	frame_req.box_max.x = max(frame_req.box_max.x,sphere.bounding_box[1].x);
	frame_req.box_max.y = max(frame_req.box_max.y,sphere.bounding_box[1].y);
	frame_req.box_max.z = max(frame_req.box_max.z,sphere.bounding_box[1].z);
}

void reshape(int width, int height) {
	// Set OpenGL view port and camera
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (double)width/height, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Send the new window size to AntTweakBar
	//TwWindowSize(width, height);
}

void updateCamera(Request &req) {
	req.cam_pos = make_float3(cos(req.cam_rot)*camera_distance,camera_height,-sin(req.cam_rot)*camera_distance);
	float3 cam_dir = -1*req.cam_pos;
	cam_dir = normalize(cam_dir);
	float3 cam_up  = make_float3(0,1,0);
	float3 cam_right = cross(cam_dir,cam_up);
	cam_right = normalize(cam_right);

	cam_up = -1*cross(cam_dir,cam_right);
	cam_up = normalize(cam_up);

	float FOV = 60.0f;
	float theta = (FOV*3.1415*0.5) / 180.0f;
	float half_width = tanf(theta);
	float aspect = (float)image_width / (float)image_height;

	float u0 = (-1*half_width) * aspect;
	float v0 = -1*half_width;
	float u1 =  half_width * aspect;
	float v1 =  half_width;
	float dist_to_image = 1;

	req.a = (u1-u0)*cam_right;
	req.b = (v1-v0)*cam_up;
	req.c = req.cam_pos + u0*cam_right + v0*cam_up + dist_to_image*cam_dir;

	if(animate)
	req.cam_rot += 0.25 * req.delta_t;
}

int find_open_dev() {
	int i;
	for (i = 0; i < 3; i++) { // Returns the world rank of the free device
		if (animation.recv_buf[i] == true) return i+1;
	}
	return 0;
}

int find_full_frame_buffer() { // Returns the index of the frame buffer to use next
	int val = ((animation.exp_frame_seq) % 3);
	if (animation.buf_size[val] != 0) return val;
	else return -1; // Returns -1 if no full framebuffers
}

void rayTrace() {
	unsigned int* out_data;
	MPI::Status status;
	Request req;
	int i;
	int buf_ind;
	int source;
	unsigned int num_pixels = IMAGE_WIDTH*IMAGE_HEIGHT;
	FrameHeader hdr;

	frame_req.type = 0;
	
	if (world_rank == 0) {
		i = find_open_dev();
		buf_ind = find_full_frame_buffer();

		if (buf_ind != -1) { // If we have a buffered frame that needs to be displayed.
			//fprintf(stderr, "Rendering buffered frame %d.\n", animation.exp_frame_seq);
			animation.buf_size[buf_ind] = 0; // Free that buffer
			buf_ind *= num_pixels; // Find the start of the frame
			unsigned int* frame = animation.frame_buf + buf_ind;
			animation.exp_frame_seq++;			// Increment expected frame
			cudaGLMapBufferObject((void**)&frame, pbo);
		} else if (i != 0) { // If we have a board thats not busy
			//fprintf(stderr, "Framebuffer empty on frame %d. Board %d is open.\n", animation.frame_seq, i);
			frame_req.frame_seq = animation.frame_seq++;
			MPI::COMM_WORLD.Send(&frame_req, sizeof(Request), MPI_CHAR, i, 0);
			animation.recv_buf[i-1] = false;

			// Data available?
			if (MPI::COMM_WORLD.Iprobe(MPI_ANY_SOURCE, 0, status)) {
				source = status.Get_source();
				MPI::COMM_WORLD.Recv(&hdr, sizeof(FrameHeader), MPI_CHAR, source, 0, status);

				// If this is the expected frame?
				if (hdr.frame_seq == animation.exp_frame_seq) {
					animation.buf_size[(hdr.frame_seq % 3)] = 0; // Set the buffer for that frame to size 0
					animation.exp_frame_seq++;	// Increment seq_num

					cudaGLMapBufferObject((void**)&out_data, pbo);
					MPI::COMM_WORLD.Recv(out_data, num_pixels, MPI_INT, source, 0, status);
			
				// No? Buffer it.
				} else {
					animation.buf_size[hdr.frame_seq % 3] = 1;
					buf_ind = (hdr.frame_seq % 3) * num_pixels;
					MPI::COMM_WORLD.Recv((animation.frame_buf+buf_ind), num_pixels, MPI_INT, source, 0, status);
				}
				
				// No matter what, open that device up for a request
				animation.recv_buf[source-1] = true;
			} 
 		} else { // If no boards are available - block until we receive something
			//fprintf(stderr, "Framebuffer empty on frame %d, and no boards are free.\n", animation.frame_seq);
			MPI::COMM_WORLD.Probe(MPI_ANY_SOURCE, 0, status);
			source = status.Get_source();
			MPI::COMM_WORLD.Recv(&hdr, sizeof(FrameHeader), MPI_CHAR, source, 0, status);

			// If this is the expected frame?
			if (hdr.frame_seq == animation.frame_seq) {
				animation.buf_size[(hdr.frame_seq % 3)] = 0;
				animation.exp_frame_seq++;

				cudaGLMapBufferObject((void**)&out_data, pbo);
				MPI::COMM_WORLD.Recv(out_data, num_pixels, MPI_INT, source, 0, status);
		
			// No? Buffer it.
			} else {
				animation.buf_size[hdr.frame_seq%3] = 1;
				buf_ind = (hdr.frame_seq % 3) * num_pixels;
				MPI::COMM_WORLD.Recv((animation.frame_buf+buf_ind), num_pixels, MPI_INT, source, 0, status);
			}

			animation.recv_buf[source-1] = true;		
		}
		cudaGLUnmapBufferObject(pbo);

	} else { // If this is slave
		MPI::COMM_WORLD.Recv(&req, sizeof(Request), MPI_CHAR, 0, 0, status);
		hdr.frame_seq = req.frame_seq;		

		if (req.type == 0) {
			RayTraceImage(frame.host_mem, image_width, image_height, total_number_of_triangles,
				req.a, req.b, req.c,
				req.cam_pos,
				req.light_pos,
				req.light_col,
				req.box_min , req.box_max);

			cudaDeviceSynchronize();
			MPI::COMM_WORLD.Send(&hdr, sizeof(FrameHeader), MPI_CHAR, 0, 0);
			MPI::COMM_WORLD.Send(frame.host_mem, image_width*image_height, MPI_INT, 0, 0);
		}
	}	

	//cudaGLUnmapBufferObject(pbo);

	if (world_rank == 0) {
		// download texture from destination PBO
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, result_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	return;
}

void displayTexture() {
	// render a screen sized quad
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, image_width, image_height);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
	//CUT_CHECK_ERROR_GL();
}

void display() {
	if (world_rank == 0) {
		
		//update the delta time for animation
		static int lastFrameTime = 0;

		if (lastFrameTime == 0) {
			lastFrameTime = glutGet(GLUT_ELAPSED_TIME);
		}

		int now = glutGet(GLUT_ELAPSED_TIME);
		int elapsedMilliseconds = now - lastFrameTime;
		frame_req.delta_t = elapsedMilliseconds / 1000.0f;
		lastFrameTime = now;

		updateCamera(frame_req);

		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	rayTrace();
	
	if (world_rank == 0) {
		displayTexture();

		//TwDraw();
		glutSwapBuffers();
		glutPostRedisplay();
	}
}

int main(int argc, char** argv) {
	int p_len;
	int size;
	char p_name[MPI_MAX_PROCESSOR_NAME];
	uint8_t *buf;
	float* dev_triangle;
	
	MPI::Init(argc, argv);

	world_size = MPI::COMM_WORLD.Get_size();
	world_rank = MPI::COMM_WORLD.Get_rank();
	MPI::Get_processor_name(p_name, p_len);
	
	std::cout << "Processor name " << p_name;
	std::cout << ", Rank " << world_rank;
	std::cout << ", Size " << world_size << std::endl;
	
	// Initialize light position
	frame_req.light_pos = make_float3(-23, 25, 3);
	frame_req.light_col = make_float3(1.0, 0.0, 0.0);
	
	if (world_rank == 0) {
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(image_width,image_height);
		glutCreateWindow("Parallel Group Raytracer");
	}

	// initialize GL
	if (world_rank == 0 && !initGL()) {
		fprintf(stderr, "\t%s, GL Initialization Failed.\n", p_name);
		MPI::Finalize();
		return 0;
	} else fprintf(stderr, "\t%s, GL Initialization Success.\n", p_name);

	// initialize CUDA
	if (!initCUDA()) {
		fprintf(stderr, "\t%s, CUDA Initialization Failed.\n", p_name);
		MPI::Finalize();
		return 0;
	} else fprintf(stderr, "\t%s, CUDA Initialization Success.\n", p_name);
	
	initCUDAmemory(dev_triangle);
	frame_req.delta_t = 0;
	animation.frame_seq = 1;
	animation.exp_frame_seq = 1;
	animation.recv_buf[0] = true; animation.buf_size[0] = 0;
	animation.recv_buf[1] = true; animation.buf_size[1] = 0;
	animation.recv_buf[2] = true; animation.buf_size[2] = 0;
	
	if (world_rank == 0) {
		// register callbacks
		glutDisplayFunc(display); // For all boards
		//glutKeyboardFunc(keyboard);
		//glutKeyboardUpFunc(KeyboardUpCallback);
		//glutSpecialUpFunc(SpecialKey);

		glutReshapeFunc(reshape);
		// - Directly redirect GLUT mouse button events to AntTweakBar
		//glutMouseFunc((GLUTmousebuttonfun)TwEventMouseButtonGLUT);
		// - Directly redirect GLUT mouse motion events to AntTweakBar
		//glutMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
		// - Directly redirect GLUT mouse "passive" motion events to AntTweakBar (same as MouseMotion)
		//glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);

		// Initialize request object, and animation object
		// start rendering main-loop
		glutMainLoop();
	}
	
	while (1) {
		display();
	}
	
	cudaThreadExit();
	
	MPI::Finalize();
	
	return 0;
}
