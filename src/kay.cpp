#include <embree3/rtcore.h>
#include <limits>
#include <pybind11/pybind11.h>
#include <random>
#include "render.h"
#include "ptr.h"
namespace py = pybind11;
 
class Rtrc
{
	public:
	bool hit_flag;
	int geo_id;
	int prim_id;
	float dist;
	

	Rtrc(bool hit,int geo,int prim,float dis):
	hit_flag(hit),geo_id(geo),prim_id(prim),dist(dis){};
};
class Rtcore
{
  public:
    RTCDevice device;
	RTCScene scene;

    Rtcore()
	{
      	device=rtcNewDevice(nullptr);
      	scene = rtcNewScene(device);
	}
	void addGeo(ptr<float> vex,ptr<unsigned int> ind,int vcnt,int fcnt)
	{
		RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		float* vertices=vex.get();
		unsigned* indices=ind.get();
 		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices,0,3 * sizeof(float), vcnt);
 		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indices, 0, 3 *  sizeof(unsigned), fcnt);
		rtcCommitGeometry(geom);
		rtcAttachGeometry(scene, geom);
 		rtcReleaseGeometry(geom);
	}
	void RTsetup(){
		rtcCommitScene(scene);
	}
	Rtrc intersect(float ox,float oy,float oz,float dx,float dy,float dz)
	{

		struct RTCIntersectContext context;
 		rtcInitIntersectContext(&context);
 		struct RTCRayHit rayhit;
		
		rayhit.ray.org_x = ox;
 		rayhit.ray.org_y = oy;
 		rayhit.ray.org_z = oz;
 		rayhit.ray.dir_x = dx;
 		rayhit.ray.dir_y = dy;
 		rayhit.ray.dir_z = dz;
 		rayhit.ray.tnear = 0;
 		rayhit.ray.tfar = std::numeric_limits<float>::infinity();
 		rayhit.ray.mask = 0;
 		rayhit.ray.flags = 0;
 		rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
 		rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
 		rtcIntersect1(scene, &context, &rayhit);

 		if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
 		{
			return Rtrc(true,rayhit.hit.geomID,rayhit.hit.primID,rayhit.ray.tfar);
  			
 		}else return Rtrc(false,0,0,0);
	}

	~Rtcore(){
		rtcReleaseScene(scene);
		rtcReleaseDevice(device);
	}
};

PYBIND11_MODULE(kay, m) {
    m.doc() = "pybind11 binding "; 

	py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<std::size_t>());
    py::class_<ptr<unsigned int>>(m, "unsigned_int_ptr")
        .def(py::init<std::size_t>());

	py::class_<Rtrc>(m,"Rtrc")
		.def(py::init<bool,int,int,float>())
		.def_readonly("hit_flag",&Rtrc::hit_flag)
		.def_readonly("geo_id",&Rtrc::geo_id)
		.def_readonly("prim_id",&Rtrc::prim_id)
		.def_readonly("dist",&Rtrc::dist);

    py::class_<Rtcore>(m,"Rtcore")
      	.def(py::init<>())
		.def("intersect",&Rtcore::intersect)//index
		.def("addGeo",&Rtcore::addGeo)
		.def("RTsetup",&Rtcore::RTsetup);
		
	m.def("render", &render, "");
}