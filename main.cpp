#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <Eigen>
#define MAX_RAY_DEPTH 5 
using namespace Eigen;

// image background color
Vector3f bgcolor(1.0f, 1.0f, 1.0f);
constexpr float kEpsilon = 1e-8;
std::vector<float> g_meshVertices;
std::vector<float> g_meshNormals;
std::vector<unsigned int> g_meshIndices;
float g_modelViewMatrix[16];
std::vector<float> cross(std::vector<float> a, std::vector<float> b)
{
	std::vector<float> result = std::vector<float>();
	result.push_back(a[1] * b[2] - b[1] * a[2]);
	result.push_back(a[2] * b[0] - b[2] * a[0]);
	result.push_back(a[0] * b[1] - b[0] * a[1]);
	/*float normalizer = sqrtf(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
	result[0] = (result[0] / normalizer);
	result[1] = (result[1] / normalizer);
	result[2] = (result[2] / normalizer);*/
	return result;

}
void computeNormals()
{
	g_meshNormals.resize(g_meshVertices.size());

	// TASK 1
	// The code below sets all normals to point in the z-axis, so we get a boring constant gray color
	// The following should be replaced with your code for normal computation
	for (int v = 0; v < g_meshIndices.size() / 3; ++v)
	{
		int k = v * 3 + 1;
		std::vector<float> a = std::vector<float>();
		std::vector<float> b = std::vector<float>();
		a.push_back(g_meshVertices[3 * g_meshIndices[k - 1]] - g_meshVertices[3 * g_meshIndices[k]]);
		a.push_back(g_meshVertices[3 * g_meshIndices[k - 1] + 1] - g_meshVertices[3 * g_meshIndices[k] + 1]);
		a.push_back(g_meshVertices[3 * g_meshIndices[k - 1] + 2] - g_meshVertices[3 * g_meshIndices[k] + 2]);

		b.push_back(g_meshVertices[3 * g_meshIndices[k]] - g_meshVertices[3 * g_meshIndices[1 + k]]);
		b.push_back(g_meshVertices[3 * g_meshIndices[k] + 1] - g_meshVertices[3 * g_meshIndices[1 + k] + 1]);
		b.push_back(g_meshVertices[3 * g_meshIndices[k] + 2] - g_meshVertices[3 * g_meshIndices[1 + k] + 2]);

		std::vector<float>normal = cross(a, b);

		g_meshNormals[3 * g_meshIndices[k]] += normal[0];
		g_meshNormals[3 * g_meshIndices[k] + 1] += normal[1];
		g_meshNormals[3 * g_meshIndices[k] + 2] += normal[2];

		g_meshNormals[3 * g_meshIndices[k + 1]] += normal[0];
		g_meshNormals[3 * g_meshIndices[k + 1] + 1] += normal[1];
		g_meshNormals[3 * g_meshIndices[k + 1] + 2] += normal[2];

		g_meshNormals[3 * g_meshIndices[k - 1]] += normal[0];
		g_meshNormals[3 * g_meshIndices[k - 1] + 1] += normal[1];
		g_meshNormals[3 * g_meshIndices[k - 1] + 2] += normal[2];


	}

	/*for (int v = 0; v < g_meshNormals.size() / 3 - 1; ++v)
	{

		g_meshNormals[3 * v] = (g_meshNormals[3 * v]+ g_meshNormals[3 * v+3])/2;
		g_meshNormals[3 * v+1] = (g_meshNormals[3 * v+1] + g_meshNormals[3 * v +1+3]) / 2;
		g_meshNormals[3 * v+2] = (g_meshNormals[3 * v+2] + g_meshNormals[3 * v +2+ 3]) / 2;


	}
	*/
	for (int v = 0; v < g_meshNormals.size() / 3; ++v)
	{
		float normalizer = sqrtf(g_meshNormals[v * 3] * g_meshNormals[v * 3] + g_meshNormals[v * 3 + 1] * g_meshNormals[v * 3 + 1] + g_meshNormals[v * 3 + 2] * g_meshNormals[v * 3 + 2]);
		g_meshNormals[v * 3] = (g_meshNormals[v * 3] / normalizer);
		g_meshNormals[v * 3 + 1] = (g_meshNormals[v * 3 + 1] / normalizer);
		g_meshNormals[v * 3 + 2] = (g_meshNormals[v * 3 + 2] / normalizer);

	}
	std::vector<float> a = std::vector<float>();
	std::vector<float> b = std::vector<float>();
	a.push_back(1.0f);
	a.push_back(2.0f);
	a.push_back(3.0f);

	b.push_back(4.0f);
	b.push_back(5.0f);
	b.push_back(6.0f);
	std::vector<float>result = cross(a, b);
	std::cout << "compute cross product" << result[0] << ", " << result[1] << ", " << result[2] << "\n";

}

void loadObj(std::string p_path)
{
	std::ifstream nfile;
	nfile.open(p_path);
	std::string s;

	while (nfile >> s)
	{
		if (s.compare("v") == 0)
		{
			float x, y, z;
			nfile >> x >> y >> z;
			g_meshVertices.push_back(x);
			g_meshVertices.push_back(y);
			g_meshVertices.push_back(z);
		}
		else if (s.compare("f") == 0)
		{
			std::string sa, sb, sc;
			unsigned int a, b, c;
			nfile >> sa >> sb >> sc;

			a = std::stoi(sa);
			b = std::stoi(sb);
			c = std::stoi(sc);

			g_meshIndices.push_back(a - 1);
			g_meshIndices.push_back(b - 1);
			g_meshIndices.push_back(c - 1);
		}
		else
		{
			std::getline(nfile, s);
		}
	}

	computeNormals();

	std::cout << p_path << " loaded. Vertices: " << g_meshVertices.size() / 3 << " Triangles: " << g_meshIndices.size() / 3 << std::endl;
}
// lights in the scene
std::vector<Vector3f> lightPositions = { Vector3f(  0.0, 60, 60)
                                       , Vector3f(-60.0, 60, 60)
                                       , Vector3f( 60.0, 60, 60) };
class Shape 
{
public:
	virtual Vector3f getsurfaceColor()const=0;
	virtual float getreflect()const=0;
	virtual float getrefract()const=0;
	virtual bool intersect(const Vector3f &rayOrigin, const Vector3f &rayDirection, float &t0, float &t1) const=0;
	virtual Vector3f getNormal(const Vector3f &RayOrigin)const=0;
};
class Triangle:public Shape
{
public:
	Vector3f vert1;
	Vector3f vert2;
	Vector3f vert3;
	Vector3f surfaceColor;
	float reflect;
	float refract;

	explicit Triangle(const Vector3f &une, const Vector3f &deux, const Vector3f &trois, const Vector3f &color, float refl, float refra ):
		vert1(une),vert2(deux),vert3(trois),surfaceColor(color),reflect(refl),refract(refra)
	{
		
	}
	Vector3f getsurfaceColor()const override
	{
		return surfaceColor;
	}
	float getreflect()const override
	{
		return reflect;
	}
	float getrefract()const override
	{
		return refract;
	}
	bool intersect(const Vector3f &rayOrigin, const Vector3f &rayDirection, float &t0, float &t1) const 
	{
		// no need to normalize
		Vector3f N = getCrossNormal(); // N 
		float denom = N.dot(N);

		// Step 1: finding P

		// check if ray and plane are parallel ?
		float NdotRayDirection = N.dot(rayDirection);
		if (fabs(NdotRayDirection) < kEpsilon) // almost 0 
			return false; // they are parallel so they don't intersect ! 

		// compute d parameter using equation 2
		float d = N.dot(vert1);

		// compute t (equation 3)
		t1 = (N.dot(rayOrigin) + d) / NdotRayDirection;
		t0 = (N.dot(rayOrigin) + d) / NdotRayDirection;
		// check if the triangle is in behind the ray
		if (t1 < 0) return false; // the triangle is behind 

		// compute the intersection point using equation 1
		Vector3f P = rayOrigin + t1 * rayDirection;

		// Step 2: inside-outside test
		Vector3f C; // vector perpendicular to triangle's plane 

		// edge 0
		C = (vert2-vert1).cross(P-vert1);
		if (N.dot(C) < 0) return false; // P is on the right side 

		// edge 1
		C = (vert3-vert2).cross(P-vert2);
		if ((N.dot(C)) < 0)  return false; // P is on the right side 

		// edge 2
		C = (vert1-vert3).cross(P-vert3);
		if ((N.dot(C)) < 0) return false; // P is on the right side; 

		//u /= denom;
		//v /= denom;

		return true; // this ray hits the triangle 
		//return false;
	}
	Vector3f getNormal(const Vector3f &RayOrigin) const override
	{
		return ((vert2 - vert1).cross(vert3 - vert1)).normalized();
	}
	Vector3f getCrossNormal()  const
	{
		return ((vert2 - vert1).cross(vert3 - vert1));
	}
};


class Sphere:public Shape
{
public:
	Vector3f center;  // position of the sphere
	float radius;  // sphere radius
	Vector3f surfaceColor; // surface color
	float reflect;
	float refract;
	
	explicit Sphere(
		const Vector3f &c,
		const float &r,
		const Vector3f &sc,float reflec, float refrac) :
		center(c), radius(r), surfaceColor(sc),reflect(reflec),refract(refrac)
	{
	}

    // line vs. sphere intersection (note: this is slightly different from ray vs. sphere intersection!)
	Vector3f getsurfaceColor()const override 
	{
		return surfaceColor;
	}
	 float getreflect()const override 
	 {
		 return reflect;
	 }
	 float getrefract()const override 
	 {
		 return refract;
	 }
	bool intersect(const Vector3f &rayOrigin, const Vector3f &rayDirection, float &t0, float &t1) const override
	{
		Vector3f l = center - rayOrigin;
		float tca = l.dot(rayDirection);
		if (tca < 0) return false;
		float d2 = l.dot(l) - tca * tca;
		if (d2 > (radius * radius)) return false;
        float thc = sqrt(radius * radius - d2);
		t0 = tca - thc;
		t1 = tca + thc;

		return true;
	}
	Vector3f getNormal(const Vector3f &RayOrigin) const override
	{
		return (RayOrigin - center).normalized();
	}
};
Vector3f mul(Vector3f a, Vector3f b) 
{
	return Vector3f( a[0] * b[0] , a[1] * b[1], a[2] * b[2]);
}
// diffuse reflection model
Vector3f diffuse(const Vector3f &L, // direction vector from the point on the surface towards a light source
	const Vector3f &N, // normal at this point on the surface
	const Vector3f &diffuseColor,
	const float kd // diffuse reflection constant
	)
{
	Vector3f resColor = Vector3f::Zero();
	resColor = 0.333*kd*fmax(L.dot(N) ,0)*diffuseColor;
	// TODO: implement diffuse shading model

	return resColor;
}

// Phong reflection model
Vector3f phong(const Vector3f &L, // direction vector from the point on the surface towards a light source
               const Vector3f &N, // normal at this point on the surface
               const Vector3f &V, // direction pointing towards the viewer
               const Vector3f &diffuseColor, 
               const Vector3f &specularColor, 
               const float kd, // diffuse reflection constant
               const float ks, // specular reflection constant
               const float alpha) // shininess constant
{
	Vector3f R = 2 * N*N.dot(L) - L;
	Vector3f resColor = Vector3f::Zero();
	resColor = 0.33*specularColor*ks*pow(fmax(R.dot(V), 0), alpha);
	resColor += diffuse(L, N, diffuseColor, kd);
	// TODO: implement Phong shading model

	return resColor;
}
float mix(const float &a, const float &b, const float &mix)
{
	return b * mix + a * (1 - mix);
}
Vector3f trace(
	const Vector3f &rayOrigin,
	const Vector3f &rayDirection,
	const std::vector<Shape*> &spheres, const int &depth)
{
	float bias = 1e-4;
	Vector3f pixelColor = Vector3f::Zero();
	float t = INFINITY;
	int index = -1;
	float t0=0;
	float t1=0;
	int size = spheres.size();
	for (int k = 0; k < size; k++)
	{
		if ((*spheres[k]).intersect(rayOrigin, rayDirection, t0, t1) == true)
		{
			if (fmin(t0, t1) < t&&fmin(t0, t1)>0)
			{
				t = fmin(t0, t1);
				index = k;
			}
			
		}
				// TODO: implement ray tracing as described in the homework description
		
	}
	//refelction/refraction
	if (index != -1 && ((*spheres[index]).getreflect() > 0 || (*spheres[index]).getrefract() > 0) && depth < MAX_RAY_DEPTH) 
	{
		pixelColor = { 0,0,0 };
		Vector3f ShadowOrigin = rayOrigin + t * rayDirection;
		Vector3f ShadowDirection;
		Vector3f normal = (*spheres[index]).getNormal(ShadowOrigin);
		Vector3f reflectDirection = (rayDirection - normal * 2 * rayDirection.dot(normal)).normalized();
		Vector3f reflection = trace(ShadowOrigin + normal * bias, reflectDirection, spheres, depth + 1);
		Vector3f refraction = { 0,0,0 };
		bool inside = false;
		int block = 0;
		for (int s = 0; s < 3; s++)
		{
			block = 0;
			ShadowDirection = (lightPositions[s] - ShadowOrigin).normalized();
			for (int k = 0; k < size; k++)
			{
				if ((*spheres[k]).intersect(ShadowOrigin, ShadowDirection, t0, t1) == true)
				{
					block = 1;
					break;
				}
			}
			if (block == 0) {

				pixelColor = pixelColor + phong(ShadowDirection, normal
					, (rayOrigin - ShadowOrigin).normalized(), (*spheres[index]).getsurfaceColor(), { 1,1,1 }, 1, 3, 100);

			}


		}
		
		if (rayDirection.dot(normal)> 0) normal = -normal, inside = true;
		float facingratio = -rayDirection.dot(normal);
		// change the mix value to tweak the effect
		
		float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
		
		if ((*spheres[index]).getrefract()==1) {
			pixelColor = { 0,0,0 };
			float ior = 1.52, eta = (inside) ? ior : 1 / ior;
			float cosi = ShadowOrigin.dot(rayDirection);
			float k = 1 - eta * eta * (1 - cosi * cosi);
			Vector3f refractionDirection = (rayDirection * eta + normal * (eta *  cosi - sqrt(k))).normalized();
			refraction = trace(ShadowOrigin-normal*bias, refractionDirection, spheres, depth + 1);
		}
		
		pixelColor += mul((
			reflection * fresneleffect +
			refraction * (1 - fresneleffect) * (*spheres[index]).getrefract()) , (*spheres[index]).getsurfaceColor());
		
		return pixelColor;
	}
	else if (index != -1) 
	{
		pixelColor = { 0,0,0 };
		Vector3f ShadowOrigin = rayOrigin + t * rayDirection;
		Vector3f ShadowDirection;
		Vector3f normal = (*spheres[index]).getNormal(ShadowOrigin);
		int block = 0;
		std::vector<int> blocksphere;
		for (int s = 0; s < 3; s++) 
		{
			block = 0;
			ShadowDirection = (lightPositions[s] - ShadowOrigin).normalized();
			for (int k = 0; k < size; k++)
			{
				if ((*spheres[k]).intersect(ShadowOrigin, ShadowDirection, t0, t1) == true)
				{
					block = 1;
					blocksphere.push_back(k);
					
				}
			}
			if (block == 0) {
				
				pixelColor = pixelColor + phong(ShadowDirection, normal
					, (rayOrigin - ShadowOrigin).normalized(), (*spheres[index]).getsurfaceColor(), {1,1,1}, 1,3,100);
				
			}
			
			if(block != 0 && blocksphere.size()==1&& (*spheres[blocksphere[0]]).getrefract()==1)
			{
				float facingratio = -rayDirection.dot(normal);
				// change the mix value to tweak the effect

				float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);

				Vector3f refraction = { 0,0,0 };
				bool inside = false;
				if (rayDirection.dot(normal) > 0) normal = -normal, inside = true;
				float ior = 1.1, eta = (inside) ? ior : 1 / ior;
				float cosi = ShadowOrigin.dot(rayDirection);
				float k = 1 - eta * eta * (1 - cosi * cosi);
				Vector3f refractionDirection = (rayDirection * eta + normal * (eta *  cosi - sqrt(k))).normalized();
				refraction = trace(ShadowOrigin - normal * bias, refractionDirection, spheres, depth + 1);
				pixelColor += mul((refraction * (1 - fresneleffect) * (*spheres[index]).getrefract()), (*spheres[index]).getsurfaceColor());
			}
			

					
		}
		
		return pixelColor;
	}
    pixelColor = bgcolor;

	
	return pixelColor;
}

void render(const std::vector<Shape*> &spheres)
{
  unsigned width = 640;
  unsigned height = 480;
  Vector3f *image = new Vector3f[width * height];
  Vector3f *pixel = image;
  float invWidth  = 1 / float(width);
  float invHeight = 1 / float(height);
  float fov = 30;
  float aspectratio = width / float(height);
	float angle = tan(M_PI * 0.5f * fov / 180.f);
	
	// Trace rays
	for (unsigned y = 0; y < height; ++y) 
	{
		for (unsigned x = 0; x < width; ++x) 
		{
			float rayX = (2 * ((x + 0.5f) * invWidth) - 1) * angle * aspectratio;
			float rayY = (1 - 2 * ((y + 0.5f) * invHeight)) * angle;
			Vector3f rayDirection(rayX, rayY, -1);
			rayDirection.normalize();
			*(pixel++) = trace(Vector3f::Zero(), rayDirection, spheres,0);
			
		}
	}
	
	// Save result to a PPM image
	std::ofstream ofs("./render.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i) 
	{
		const float x = image[i](0);
		const float y = image[i](1);
		const float z = image[i](2);

		ofs << (unsigned char)(std::min(float(1), x) * 255) 
			  << (unsigned char)(std::min(float(1), y) * 255) 
			  << (unsigned char)(std::min(float(1), z) * 255);
	}
	
	ofs.close();
	delete[] image;
}

int main(int argc, char **argv)
{
	std::vector<Shape*> spheres;
	// position, radius, surface color, reflection:value refraction: t/f 
	//background
	spheres.push_back(new Sphere(Vector3f(0.0, -10004, -20), 10000, Vector3f(0.50, 0.50, 0.50),0,0));
	//actual shperes
	//spheres.push_back(new Sphere(Vector3f(0.0, 0, -20), 4, Vector3f(1.00, 0.32, 0.36),0.5,1));//red
	//spheres.push_back(new Sphere(Vector3f(5.0, -1, -15), 2, Vector3f(0.90, 0.76, 0.46),0.5,1));//yellow
	//spheres.push_back(new Sphere(Vector3f(5.0, 0, -25), 3, Vector3f(0.65, 0.77, 0.97),0.5,1));//blue
	spheres.push_back(new Sphere(Vector3f(-5.5, 0, -13), 3, Vector3f(0.90, 0.90, 0.90),0,0));//white
	spheres.push_back(new Triangle(Vector3f(5.0, -1, -15), Vector3f(0.0, 0, -20), Vector3f(5.0, 0, -25), Vector3f(1.00, 0.32, 0.36), 0.5, 0));//white
	loadObj("teapot.obj");

	render(spheres);

	return 0;
}
