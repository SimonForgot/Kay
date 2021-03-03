#pragma once
#include "ptr.h"
#include <embree3/rtcore.h>
class Rtrc {
public:
  bool hit_flag;
  int geo_id;
  int prim_id;
  float dist;

  Rtrc(bool hit, int geo, int prim, float dis)
      : hit_flag(hit), geo_id(geo), prim_id(prim), dist(dis){};
};

class Rtcore {
public:
  RTCDevice device;
  RTCScene scene;

  Rtcore() {
    device = rtcNewDevice(nullptr);
    scene = rtcNewScene(device);
  }

  ~Rtcore() {
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
  }

  void addGeo(ptr<float> vex, ptr<unsigned int> ind, int vcnt, int fcnt);
  void RTsetup() { rtcCommitScene(scene); }
  Rtrc intersect(float ox, float oy, float oz, float dx, float dy, float dz);
};