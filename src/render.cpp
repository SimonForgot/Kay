#include <random>
#include "ptr.h"
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
using Real = float;

// some basic vector operations
template <typename T>
struct Vec2
{
    T x, y;
    Vec2(T x = 0, T y = 0) : x(x), y(y) {}
};
template <typename T>
struct Vec3
{
    T x, y, z;
    Vec3(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}
};
using Vec2f = Vec2<Real>;
using Vec3i = Vec3<int>;
using Vec3f = Vec3<Real>;
Vec2f operator+(const Vec2f &v0, const Vec2f &v1) { return Vec2f{v0.x + v1.x, v0.y + v1.y}; }
Vec2f &operator+=(Vec2f &v0, const Vec2f &v1)
{
    v0.x += v1.x;
    v0.y += v1.y;
    return v0;
}
Vec2f operator-(const Vec2f &v0, const Vec2f &v1) { return Vec2f{v0.x - v1.x, v0.y - v1.y}; }
Vec2f operator*(Real s, const Vec2f &v) { return Vec2f{v.x * s, v.y * s}; }
Vec2f operator*(const Vec2f &v, Real s) { return Vec2f{v.x * s, v.y * s}; }
Vec2f operator/(const Vec2f &v, Real s) { return Vec2f{v.x / s, v.y / s}; }
Real dot(const Vec2f &v0, const Vec2f &v1) { return v0.x * v1.x + v0.y * v1.y; }
Real length(const Vec2f &v) { return sqrt(dot(v, v)); }
Vec2f normal(const Vec2f &v) { return Vec2f{-v.y, v.x}; }
Vec3f &operator+=(Vec3f &v0, const Vec3f &v1)
{
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}
Vec3f operator-(const Vec3f &v0, const Vec3f &v1) { return Vec3f{v0.x - v1.x, v0.y - v1.y, v0.z - v1.z}; }
Vec3f operator-(const Vec3f &v) { return Vec3f{-v.x, -v.y, -v.z}; }
Vec3f operator*(const Vec3f &v, Real s) { return Vec3f{v.x * s, v.y * s, v.z * s}; }
Vec3f operator*(Real s, const Vec3f &v) { return Vec3f{v.x * s, v.y * s, v.z * s}; }
Vec3f operator/(const Vec3f &v, Real s) { return Vec3f{v.x / s, v.y / s, v.z / s}; }
Real dot(const Vec3f &v0, const Vec3f &v1) { return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z; }
template <typename T>
T clamp(T v, T l, T u)
{
    if (v < l)
        return l;
    else if (v > u)
        return u;
    return v;
}
struct TriangleMesh
{
    std::vector<Vec2f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors; // defined for each face
};

Vec3f raytrace(const TriangleMesh &mesh,
               const Vec2f &screen_pos,
               int *hit_index = nullptr)
{
    // loop over all triangles in a mesh, return the first one that hits
    for (int i = 0; i < (int)mesh.indices.size(); i++)
    {
        // retrieve the three vertices of a triangle
        auto index = mesh.indices[i];
        auto v0 = mesh.vertices[index.x], v1 = mesh.vertices[index.y], v2 = mesh.vertices[index.z];
        // form three half-planes: v1-v0, v2-v1, v0-v2
        // if a point is on the same side of all three half-planes, it's inside the triangle.
        auto n01 = normal(v1 - v0), n12 = normal(v2 - v1), n20 = normal(v0 - v2);
        auto side01 = dot(screen_pos - v0, n01) > 0;
        auto side12 = dot(screen_pos - v1, n12) > 0;
        auto side20 = dot(screen_pos - v2, n20) > 0;
        if ((side01 && side12 && side20) || (!side01 && !side12 && !side20))
        {
            if (hit_index != nullptr)
            {
                *hit_index = i;
            }
            return mesh.colors[i];
        }
    }
    // return background
    if (hit_index != nullptr)
    {
        *hit_index = -1;
    }
    return Vec3f{0, 0, 0};
}
void render(ptr<float> shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            ptr<float> color,
            ptr<float> rendered_image)
{
    int pic_res = 256;
    TriangleMesh mesh;
    for (auto i = 0; i < p_num; i++)
        mesh.vertices.push_back({shape[2 * i], shape[2 * i + 1]});
    for (auto i = 0; i < tri_num; i++)
        mesh.indices.push_back({indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]});
    for (auto i = 0; i < tri_num; i++)
        mesh.colors.push_back({color[3 * i], color[3 * i + 1], color[3 * i + 2]});

    std::uniform_real_distribution<float> uni_dist(0, 1);
    std::mt19937 rng(1234);
    int samples_per_pixel = 4;
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    for (int y = 0; y < pic_res; y++)
    { // for each pixel
        for (int x = 0; x < pic_res; x++)
        {
            for (int dy = 0; dy < sqrt_num_samples; dy++)
            { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++)
                {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{x + xoff, y + yoff};
                    auto color = raytrace(mesh, screen_pos);
                    auto temp = color / samples_per_pixel;
                    rendered_image[3 * (y * pic_res + x)] += temp.x;
                    rendered_image[3 * (y * pic_res + x) + 1] += temp.y;
                    rendered_image[3 * (y * pic_res + x) + 2] += temp.z;
                }
            }
        }
    }
}
struct Edge
{
    int v0, v1; // vertex ID, v0 < v1

    Edge(int v0, int v1) : v0(std::min(v0, v1)), v1(std::max(v0, v1)) {}

    // for sorting edges
    bool operator<(const Edge &e) const
    {
        return this->v0 != e.v0 ? this->v0 < e.v0 : this->v1 < e.v1;
    }
};
struct Sampler
{
    std::vector<Real> pmf, cdf;
};
struct Img
{
    Img(int width, int height, const Vec3f &val = Vec3f{0, 0, 0}) : width(width), height(height)
    {
        color.resize(width * height, val);
    }

    std::vector<Vec3f> color;
    int width;
    int height;
};

// build a discrete CDF using edge length
Sampler build_edge_sampler(const TriangleMesh &mesh,
                           const std::vector<Edge> &edges)
{
    std::vector<Real> pmf;
    std::vector<Real> cdf;
    pmf.reserve(edges.size());
    cdf.reserve(edges.size() + 1);
    cdf.push_back(0);
    for (auto edge : edges)
    {
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        pmf.push_back(length(v1 - v0));
        cdf.push_back(pmf.back() + cdf.back());
    }
    auto length_sum = cdf.back();
    for_each(pmf.begin(), pmf.end(), [&](Real &p) { p /= length_sum; });
    for_each(cdf.begin(), cdf.end(), [&](Real &p) { p /= length_sum; });
    return Sampler{pmf, cdf};
}

// binary search for inverting the CDF in the sampler
int sample(const Sampler &sampler, const Real u)
{
    auto cdf = sampler.cdf;
    return clamp<int>(upper_bound(
                          cdf.begin(), cdf.end(), u) -
                          cdf.begin() - 1,
                      0, cdf.size() - 2);
}

// given a triangle mesh, collect all edges.
std::vector<Edge> collect_edges(const TriangleMesh &mesh)
{
    std::set<Edge> edges;
    for (auto index : mesh.indices)
    {
        edges.insert(Edge(index.x, index.y));
        edges.insert(Edge(index.y, index.z));
        edges.insert(Edge(index.z, index.x));
    }
    return std::vector<Edge>(edges.begin(), edges.end());
}

struct DTriangleMesh
{
    DTriangleMesh(int num_vertices)
    {
        vertices.resize(num_vertices, Vec2f{0, 0});
    }
    std::vector<Vec2f> vertices;
};

void compute_edge_derivatives(
    const TriangleMesh &mesh,
    const std::vector<Edge> &edges,
    const Sampler &edge_sampler,
    const Img &adjoint,
    const int num_edge_samples,
    std::mt19937 &rng,
    std::vector<Vec2f> &d_vertices)
{

    std::uniform_real_distribution<float> uni_dist(0, 1);

    for (int i = 0; i < num_edge_samples; i++)
    {
        // pick an edge

        auto edge_id = sample(edge_sampler, uni_dist(rng));
        auto edge = edges[edge_id];
        auto pmf = edge_sampler.pmf[edge_id];
        // pick a point p on the edge
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        auto t = uni_dist(rng);
        auto p = v0 + t * (v1 - v0);
        auto xi = (int)p.x;
        auto yi = (int)p.y; // integer coordinates
        if (xi < 0 || yi < 0 || xi >= adjoint.width || yi >= adjoint.height)
        {
            continue;
        }

        // sample the two sides of the edge
        auto n = normal((v1 - v0) / length(v1 - v0));
        auto color_in = raytrace(mesh, p - 1e-3f * n);
        auto color_out = raytrace(mesh, p + 1e-3f * n);

        // get corresponding adjoint from the adjoint image,
        // multiply with the color difference and divide by the pdf & number of samples.
        auto pdf = pmf / (length(v1 - v0));
        auto weight = Real(1 / (pdf * Real(num_edge_samples)));
        auto adj = dot(color_in - color_out, adjoint.color[yi * adjoint.width + xi]);

        auto d_v0 = Vec2f{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
        auto d_v1 = Vec2f{t * n.x, t * n.y} * adj * weight;

        d_vertices[edge.v0] += d_v0;
        d_vertices[edge.v1] += d_v1;
    }
}

void d_render(ptr<float> shape,
              int p_num,
              ptr<unsigned int> indices,
              int tri_num,
              ptr<float> color,
              ptr<float> grad_img,
              ptr<float> d_shape)
{
    int pic_res = 256;
    TriangleMesh mesh;
    for (auto i = 0; i < p_num; i++)
        mesh.vertices.push_back({shape[2 * i], shape[2 * i + 1]});
    for (auto i = 0; i < tri_num; i++)
        mesh.indices.push_back({indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]});
    for (auto i = 0; i < tri_num; i++)
        mesh.colors.push_back({color[3 * i], color[3 * i + 1], color[3 * i + 2]});

    Img adjoint(pic_res, pic_res, Vec3f{1, 1, 1});
    DTriangleMesh d_mesh(mesh.vertices.size());
    auto edges = collect_edges(mesh);
    auto edge_sampler = build_edge_sampler(mesh, edges);

    std::mt19937 rng(1234);
    for (auto i = 0; i < adjoint.width * adjoint.height; i++)
    {
        adjoint.color[i] = Vec3f{grad_img[3 * i],
                                 grad_img[3 * i + 1],
                                 grad_img[3 * i + 2]};
    }
    compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, 256 * 256,
                             rng, d_mesh.vertices);

    int s = 0;
    for (auto i : d_mesh.vertices)
    {
        d_shape[s++] = i.x;
        d_shape[s++] = i.y;
    }
}