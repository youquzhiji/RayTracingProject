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

// lights in the scene
std::vector<Vector3f> lightPositions = { Vector3f(  0.0, 60, 60)
                                       , Vector3f(-60.0, 60, 60)
                                       , Vector3f( 60.0, 60, 60) };

class Sphere
{
public:
	Vector3f center;  // position of the sphere
	float radius;  // sphere radius
	Vector3f surfaceColor; // surface color
	float reflect;
	float refract;
	
	Sphere(
		const Vector3f &c,
		const float &r,
		const Vector3f &sc,float reflec, float refrac) :
		center(c), radius(r), surfaceColor(sc),reflect(reflec),refract(refrac)
	{
	}

    // line vs. sphere intersection (note: this is slightly different from ray vs. sphere intersection!)
	bool intersect(const Vector3f &rayOrigin, const Vector3f &rayDirection, float &t0, float &t1) const
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
	const std::vector<Sphere> &spheres, const int &depth)
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
		if (spheres[k].intersect(rayOrigin, rayDirection, t0, t1) == true)
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
	if (index != -1 && (spheres[index].reflect > 0 || spheres[index].refract > 0) && depth < MAX_RAY_DEPTH) 
	{
		pixelColor = { 0,0,0 };
		Vector3f ShadowOrigin = rayOrigin + t * rayDirection;
		Vector3f ShadowDirection;
		Vector3f normal = (ShadowOrigin - spheres[index].center).normalized();
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
				if (spheres[k].intersect(ShadowOrigin, ShadowDirection, t0, t1) == true)
				{
					block = 1;
					break;
				}
			}
			if (block == 0) {

				pixelColor = pixelColor + phong(ShadowDirection, normal
					, (rayOrigin - ShadowOrigin).normalized(), spheres[index].surfaceColor, { 1,1,1 }, 1, 3, 100);

			}


		}
		
		if (rayDirection.dot(normal)> 0) normal = -normal, inside = true;
		float facingratio = -rayDirection.dot(normal);
		// change the mix value to tweak the effect
		
		float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
		
		if (spheres[index].refract==1) {
			pixelColor = { 0,0,0 };
			float ior = 1.1, eta = (inside) ? ior : 1 / ior;
			float cosi = ShadowOrigin.dot(rayDirection);
			float k = 1 - eta * eta * (1 - cosi * cosi);
			Vector3f refractionDirection = (rayDirection * eta + normal * (eta *  cosi - sqrt(k))).normalized();
			refraction = trace(ShadowOrigin-normal*bias, refractionDirection, spheres, depth + 1);
		}
		
		pixelColor += mul((
			reflection * fresneleffect +
			refraction * (1 - fresneleffect) * spheres[index].refract) , spheres[index].surfaceColor);
		
		return pixelColor;
	}
	else if (index != -1) 
	{
		pixelColor = { 0,0,0 };
		Vector3f ShadowOrigin = rayOrigin + t * rayDirection;
		Vector3f ShadowDirection;
		Vector3f normal = (ShadowOrigin - spheres[index].center).normalized();
		int block = 0;
		for (int s = 0; s < 3; s++) 
		{
			block = 0;
			ShadowDirection = (lightPositions[s] - ShadowOrigin).normalized();
			for (int k = 0; k < size; k++)
			{
				if (spheres[k].intersect(ShadowOrigin, ShadowDirection, t0, t1) == true)
				{
					block = 1;
					break;
				}
			}
			if (block == 0) {
				
				pixelColor = pixelColor + phong(ShadowDirection, normal
					, (rayOrigin - ShadowOrigin).normalized(), spheres[index].surfaceColor, {1,1,1}, 1,3,100);
				
			}

					
		}
		
		return pixelColor;
	}
    pixelColor = bgcolor;

	
	return pixelColor;
}

void render(const std::vector<Sphere> &spheres)
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
	std::vector<Sphere> spheres;
	// position, radius, surface color, reflection:value refraction: t/f 
	//background
	spheres.push_back(Sphere(Vector3f(0.0, -10004, -20), 10000, Vector3f(0.50, 0.50, 0.50),0,0));
	//actual shperes
	spheres.push_back(Sphere(Vector3f(0.0, 0, -20), 4, Vector3f(1.00, 0.32, 0.36),0.5,0));//red
	spheres.push_back(Sphere(Vector3f(5.0, -1, -15), 2, Vector3f(0.90, 0.76, 0.46),0.5,1));//yellow
	spheres.push_back(Sphere(Vector3f(5.0, 0, -25), 3, Vector3f(0.65, 0.77, 0.97),0.5,0));//blue
	spheres.push_back(Sphere(Vector3f(-5.5, 0, -13), 3, Vector3f(0.90, 0.90, 0.90),0,0));//white

	render(spheres);

	return 0;
}
