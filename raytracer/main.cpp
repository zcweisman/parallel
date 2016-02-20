#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


#include <AntTweakBar.h>
#include "utils.h"
#include "cutil_math.h"

using namespace std;

// Globals ---------------------------------------
unsigned int window_width  = 800;
unsigned int window_height = 600;
unsigned int image_width   = 800;
unsigned int image_height  = 600;
float delta_t = 0;

GLuint pbo;               // this pbo is used to connect CUDA and openGL
GLuint result_texture;    // the ray-tracing result is copied to this openGL texture
TriangleMesh mesh;

TriangleMesh ground;
TriangleMesh sphere;
TriangleMesh object;
int total_number_of_triangles = 0;

float *dev_triangle_p; // the cuda device pointer that points to the uploaded triangles


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

bool initGL();
bool initCUDA( int argc, char **argv);
void initCUDAmemory();
void Terminate(void);
void initTweakMenus();
void display();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void KeyboardUpCallback(unsigned char key, int x, int y);
void SpecialKey(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void rayTrace();

TwBar *bar; // Pointer to the tweak bar

void initTweakMenus() {
	if( !TwInit(TW_OPENGL, NULL) )
	{
		// A fatal error occurred
		fprintf(stderr, "AntTweakBar initialization failed: %s\n", TwGetLastError());
		exit(0);
	}

	bar = TwNewBar("Parameters");

	TwAddVarRW(bar, "camera rotation", TW_TYPE_FLOAT, &camera_rotation,
		" min=-5.0 max=5.0 step=0.01 group='Camera'");
	TwAddVarRW(bar, "camera distance", TW_TYPE_FLOAT, &camera_distance,
		" min= 1.0 max=125.0 step=0.1 group='Camera'");
	TwAddVarRW(bar, "camera height", TW_TYPE_FLOAT, &camera_height,
		" min= -35.0 max= 100.0 step=0.1 group='Camera'");

	TwAddVarRW(bar, "light_pos_x", TW_TYPE_FLOAT, &light_x,
		" min= -100.0 max= 100.0 step=0.1 group='Light_source'");
	TwAddVarRW(bar, "light_pos_y", TW_TYPE_FLOAT, &light_y,
		" min= -100.0 max= 100.0 step=0.1 group='Light_source'");
	TwAddVarRW(bar, "light_pos_z", TW_TYPE_FLOAT, &light_z,
		" min= -100.0 max= 100.0 step=0.1 group='Light_source'");

	TwAddVarRW(bar,"light_color",TW_TYPE_COLOR3F, &light_color, " group='Light_source' ");
}

void Terminate(void) {
	TwTerminate();
}

bool initGL() {
	glewInit();
	if (! glewIsSupported
		(
		"GL_VERSION_4_1 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		))
	{
			fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
			fflush(stderr);
			//return CUTFalse;
            return false;
	}

	// init openGL state
	glClearColor(0, 0, 0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// view-port
	glViewport(0, 0, window_width, window_height);

	initTweakMenus();
	return true;
}

bool initCUDA( int argc, char **argv) {
    cudaGLSetGLDevice(0);

	return true;
}

void initCUDAmemory() {
	// initialize the PBO for transferring data from CUDA to openGL
	unsigned int num_texels = image_width * image_height;
	unsigned int size_tex_data = sizeof(GLubyte) * num_texels * 4;
	void *data = malloc(size_tex_data);

	// create buffer object
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

	cout << "total number of triangles check:" << mesh.faces.size() + sphere.faces.size() << " == " << triangles.size()/3 << endl;

	size_t triangle_size = triangles.size() * sizeof(float4);
	total_number_of_triangles = triangles.size()/3;

	if(triangle_size > 0) {
		cudaMalloc((void **)&dev_triangle_p, triangle_size);
		cudaMemcpy(dev_triangle_p,&triangles[0],triangle_size,cudaMemcpyHostToDevice);
		bindTriangles(dev_triangle_p, total_number_of_triangles);
	}

	scene_aabbox_min = mesh.bounding_box[0];
	scene_aabbox_max = mesh.bounding_box[1];

	scene_aabbox_min.x = min(scene_aabbox_min.x,sphere.bounding_box[0].x);
	scene_aabbox_min.y = min(scene_aabbox_min.y,sphere.bounding_box[0].y);
	scene_aabbox_min.z = min(scene_aabbox_min.z,sphere.bounding_box[0].z);

	scene_aabbox_max.x = max(scene_aabbox_max.x,sphere.bounding_box[1].x);
	scene_aabbox_max.y = max(scene_aabbox_max.y,sphere.bounding_box[1].y);
	scene_aabbox_max.z = max(scene_aabbox_max.z,sphere.bounding_box[1].z);
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
	TwWindowSize(width, height);
}

void SpecialKey(int key, int x, int y) {
	switch(key)	{
	       case GLUT_KEY_F1:
		         break;
    };
}

void updateCamera() {
	campos = make_float3(cos(camera_rotation)*camera_distance,camera_height,-sin(camera_rotation)*camera_distance);
	float3 cam_dir = -1*campos;
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

	a = (u1-u0)*cam_right;
	b = (v1-v0)*cam_up;
	c = campos + u0*cam_right + v0*cam_up + dist_to_image*cam_dir;

	if(animate)
	camera_rotation += 0.25 * delta_t;
}

void rayTrace() {
	unsigned int* out_data;

	cudaGLMapBufferObject( (void**)&out_data, pbo);

	RayTraceImage(out_data, image_width, image_height, total_number_of_triangles,
		a, b, c,
		campos,
		make_float3(light_x,light_y,light_z),
		make_float3(light_color[0],light_color[1],light_color[2]),
		scene_aabbox_min , scene_aabbox_max);

	cudaGLUnmapBufferObject( pbo);

	// download texture from destination PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, result_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void displayTexture() {
	// render a screen sized quad
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glDepthMask(GL_TRUE);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, window_width, window_height);

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
	//update the delta time for animation
	static int lastFrameTime = 0;

	if (lastFrameTime == 0) {
		lastFrameTime = glutGet(GLUT_ELAPSED_TIME);
	}

	int now = glutGet(GLUT_ELAPSED_TIME);
	int elapsedMilliseconds = now - lastFrameTime;
	delta_t = elapsedMilliseconds / 1000.0f;
	lastFrameTime = now;

	updateCamera();

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	rayTrace();
	displayTexture();

	TwDraw();

	glutSwapBuffers();
	glutPostRedisplay();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch(key) {
	case ' ':
		animate = !animate;
		break;
	case(27) :
		Terminate();
		exit(0);
	}
}

void KeyboardUpCallback(unsigned char key, int x, int y) {
	if(TwEventKeyboardGLUT(key,x, y)) {
		return;
	}
}

int main(int argc, char** argv) {
	// Create GL context
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width,window_height);
	glutCreateWindow("Parallel Group Raytracer");

	// initialize GL
    if (!initGL())
		return 0;

	// initialize CUDA
    if (!initCUDA(argc, argv))
		return 0;

	initCUDAmemory();

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(KeyboardUpCallback);
	glutSpecialUpFunc(SpecialKey);

	glutReshapeFunc(reshape);
	// - Directly redirect GLUT mouse button events to AntTweakBar
	glutMouseFunc((GLUTmousebuttonfun)TwEventMouseButtonGLUT);
	// - Directly redirect GLUT mouse motion events to AntTweakBar
	glutMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
	// - Directly redirect GLUT mouse "passive" motion events to AntTweakBar (same as MouseMotion)
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);

	// start rendering main-loop
	glutMainLoop();
	cudaThreadExit();

	return 0;
}
