#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "types.h"

using namespace std;

extern "C" void RayTraceImage(unsigned int*, int, int, int, float3, float3, float3,
                                float3, float3, float3, float3, float3);

extern "C" void bindTriangles(float*, unsigned int);

void loadObj(const std::string filename, TriangleMesh &mesh, int scale) {
    std::ifstream in(filename.c_str());

    if(!in.good())
    {
        cout  << "ERROR: loading obj:(" << filename << ") file is not good" << "\n";
        exit(0);
    }

    char buffer[256], str[255];
    float f1,f2,f3;

    while(!in.getline(buffer,255).eof())
    {
        buffer[255]='\0';

        //sscanf_s(buffer,"%s",str,255);
        memcpy(str, buffer, 255);

        // reading a vertex
        if (buffer[0]=='v' && (buffer[1]==' '  || buffer[1]==32) )
        {
            if ( sscanf(buffer,"v %f %f %f",&f1,&f2,&f3)==3)
            {
                mesh.verts.push_back(make_float3(f1*scale,f2*scale,f3*scale));
            }
            else
            {
                cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
                exit(-1);
            }
        }
        // reading FaceMtls
        else if (buffer[0]=='f' && (buffer[1]==' ' || buffer[1]==32) )
        {
            TriangleFace f;
            int nt = sscanf(buffer,"f %d %d %d",&f.v[0],&f.v[1],&f.v[2]);
            if( nt!=3 )
            {
                cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
                exit(-1);
            }

            mesh.faces.push_back(f);
        }
    }

    // calculate the bounding box
    mesh.bounding_box[0] = make_float3(1000000,1000000,1000000);
    mesh.bounding_box[1] = make_float3(-1000000,-1000000,-1000000);
    for(unsigned int i = 0; i < mesh.verts.size(); i++)
    {
        //update min value
        mesh.bounding_box[0].x = min(mesh.verts[i].x,mesh.bounding_box[0].x);
        mesh.bounding_box[0].y = min(mesh.verts[i].y,mesh.bounding_box[0].y);
        mesh.bounding_box[0].z = min(mesh.verts[i].z,mesh.bounding_box[0].z);

        //update max value
        mesh.bounding_box[1].x = max(mesh.verts[i].x,mesh.bounding_box[1].x);
        mesh.bounding_box[1].y = max(mesh.verts[i].y,mesh.bounding_box[1].y);
        mesh.bounding_box[1].z = max(mesh.verts[i].z,mesh.bounding_box[1].z);

    }

    cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << endl;
    cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z <<") max:"
        << mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z <<")" << endl;
}
