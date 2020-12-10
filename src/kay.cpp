#include <embree3/rtcore.h>
#include <limits>
#include <pybind11/pybind11.h>
#include "ptr.h"
namespace py = pybind11;
 
class Rtrc
{
	public:
	bool hit_flag;
	int geo_id;
	float dist;

	Rtrc(bool hit,int geo,float dis):
	hit_flag(hit),geo_id(geo),dist(dis){};
};
class Rtcore
{
  public:
    RTCDevice device;
	RTCScene scene;
	RTCGeometry geom;
	float* vertices;
	unsigned* indices;
	
    Rtcore(ptr<float> vex,ptr<unsigned int> ind)
	{
      	device=rtcNewDevice(nullptr);
      	scene = rtcNewScene(device);
 		geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		float* vertices=vex.get();
		unsigned* indices=ind.get();
 		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices,0,3 * sizeof(float), 3);
 		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indices, 0, 3 * sizeof(unsigned), 1);
 		rtcCommitGeometry(geom);
 		rtcAttachGeometry(scene, geom);
 		rtcReleaseGeometry(geom);
 		rtcCommitScene(scene);
	}

	Rtrc intersect()
	{
		struct RTCIntersectContext context;
 		rtcInitIntersectContext(&context);
 		struct RTCRayHit rayhit;
		rayhit.ray.org_x = 0;
 		rayhit.ray.org_y = 0;
 		rayhit.ray.org_z = -19;
 		rayhit.ray.dir_x = 0;
 		rayhit.ray.dir_y = 0;
 		rayhit.ray.dir_z = 1;
 		rayhit.ray.tnear = 0;
 		rayhit.ray.tfar = std::numeric_limits<float>::infinity();
 		rayhit.ray.mask = 0;
 		rayhit.ray.flags = 0;
 		rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
 		rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
 		rtcIntersect1(scene, &context, &rayhit);

 		if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
 		{
			return Rtrc(true,rayhit.hit.geomID,rayhit.ray.tfar);
  			
 		}else return Rtrc(false,0,0);
	}

	~Rtcore(){
		rtcReleaseScene(scene);
		rtcReleaseDevice(device);
	}
};



PYBIND11_MODULE(kay, m) {
    m.doc() = "Embree raytracing core pybind11 binding "; 

	py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<unsigned int>>(m, "unsigned_int_ptr")
        .def(py::init<std::size_t>());


	py::class_<Rtrc>(m,"Rtrc")
		.def(py::init<bool,int,float>())
		.def_readonly("hit_flag",&Rtrc::hit_flag)
		.def_readonly("geo_id",&Rtrc::geo_id)
		.def_readonly("dist",&Rtrc::dist);

    py::class_<Rtcore>(m,"Rtcore")
      	.def(py::init<ptr<float>,//vertex
	  				ptr<unsigned int>>())
		.def("intersect",&Rtcore::intersect);//index
}