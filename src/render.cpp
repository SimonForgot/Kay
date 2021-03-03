#include"render.h"
#include <cmath>
#include<stdio.h>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include "kay.h"
#include"kmath.h"

struct TriangleMesh
{
    std::vector<Vec2f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors; // defined for each face
};


void render(ptr<float> mv_shape,
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            float f_dist,
             int pic_res,
             ptr<float> colors,
            ptr<float> rendered_image)
{
    float shape[3*p_num];
    float *p_mv_shape=mv_shape.get();
    for(auto i=0;i<p_num;i++)
    {
        shape[3*i]=p_mv_shape[i];
        shape[3*i+1]=p_mv_shape[i+p_num];
        shape[3*i+2]=p_mv_shape[i+2*p_num];
        //printf("pos:%f  %f  %f\n",shape[3*i],shape[3*i+1],shape[3*i+2]);
    }

    unsigned int*p_indices=indices.get();
    unsigned indix[3*tri_num];
    float color[3*tri_num];
    for(auto i=0;i<tri_num;i++)
    {
        indix[3*i]=p_indices[6*i];
        indix[3*i+1]=p_indices[6*i+2];
        indix[3*i+2]=p_indices[6*i+4];
        //printf("%u %u %u\n", indix[3*i], indix[3*i+1], indix[3*i+2]);
        color[3*i]=colors[3*i];
        color[3*i+1]=colors[3*i+1];
        color[3*i+2]=colors[3*i+2];
    }

   
    //embree render setup
    Rtcore rt{};
    rt.addGeo(shape, indix, p_num, tri_num);
    rt.RTsetup();
    
    std::uniform_real_distribution<float> uni_dist(0, 1);
    std::mt19937 rng(1234);
    int samples_per_pixel = 4;
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    for (int y = 0; y <pic_res; y++)
    { // for each pixel
        for (int x = 0; x < pic_res; x++)
        {
            for (int dy = 0; dy < sqrt_num_samples; dy++)
            { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++)
                {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{x + xoff-(pic_res/2), y + yoff-(pic_res/2)};
                    
                    auto dir=normalize(Vec3f{screen_pos.x,screen_pos.y,-f_dist});//
                    //printf("%f %f %f\n",screen_pos.x,screen_pos.y,-f_dist);
                    Rtrc record= rt.intersect(0,0,0, dir.x,dir.y,dir.z);
                    auto idx=record.prim_id;
                    
                    auto p_color = record.hit_flag?Vec3f{color[3*idx],color[3*idx+1],color[3*idx+2]}:Vec3f{0,0,0};
                   
                    auto temp = p_color / samples_per_pixel;
                    auto id=3*((pic_res-y)* pic_res + x); 
                   
                    rendered_image[ id] += temp.x;
                    rendered_image[id+ 1] += temp.y;
                    rendered_image[id+ 2] += temp.z;
                }
            }
        }
    }
}
struct Edge
{
    int v0, v1; // vertex ID, v0 < v1
    bool view_n;
    bool light_n;
    Edge(int v0, int v1,bool vn=false,bool ln=false) : v0(std::min(v0, v1)), v1(std::max(v0, v1)),view_n(vn),light_n(ln) {}
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
        int mul=10;
        if(edge.view_n)mul=2*mul;
        if(edge.light_n)mul=2*mul;
        //switch mul 531
        //mul=1;
        pmf.push_back(mul*length(v1 - v0));
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
std::vector<Edge> collect_edges(const TriangleMesh &mesh,ptr<float> normals)
{
    std::set<Edge> edges;
    int i=0;
    for (auto index : mesh.indices)
    {
        bool vn=false,ln=false;
        Vec3f n=Vec3f{normals[3*i], normals[3*i+1], normals[3*i+2]};
        if(dot(n,Vec3f{0,0,1})>0.5)vn=true;
        auto temp=dot(n,Vec3f{0,1,0});
        if(temp>0&&temp<0.5)ln=true;
        edges.insert(Edge(index.x, index.y,vn,ln));
        edges.insert(Edge(index.y, index.z,vn,ln));
        edges.insert(Edge(index.z, index.x,vn,ln));
        i++;
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
    float *shape,//mv_shape
    float f_dist,
    int pic_res,
    const std::vector<Edge> &edges,
    const Sampler &edge_sampler,
    const Img &adjoint,
    const int num_edge_samples,
    std::mt19937 &rng,
    std::vector<Vec2f> &d_vertices)
{
    std::uniform_real_distribution<float> uni_dist(0, 1);
    int p_num=mesh.vertices.size();
    int f_num=mesh.indices.size();
    //std::cout<<p_num<<" "<<f_num<<"\n";
    //for(auto i=0;i<9;i++)
    //std::cout<<shape[i]<<" ";
    //std::cout<<"\n";
    
    unsigned indix[3*f_num];
    for(auto i=0;i<f_num;i++)
    {
        indix[3*i]=mesh.indices[i].x;
        indix[3*i+1]=mesh.indices[i].y;
        indix[3*i+2]=mesh.indices[i].z;
        //std::cout<<indix[3*i]<<" "<<indix[3*i+1]<<" "<<indix[3*i+2]<<"\n";
    }
    Rtcore rt{};
    rt.addGeo(shape, indix, p_num, f_num);
    rt.RTsetup();
    //num_edge_samples
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

        //std::cout<<v0.x<<" "<<v0.y<<"\n";
        //std::cout<<v1.x<<" "<<v1.y<<"\n";
        //std::cout<<p.x<<" "<<p.y<<"\n\n";
        auto xi = (int)p.x;
        auto yi = (int)p.y; // integer coordinates
        if (xi < 0 || yi < 0 || xi >= adjoint.width || yi >= adjoint.height)
        {
            continue;
        }
        // sample the two sides of the edge
        auto n = normal((v1 - v0) / length(v1 - v0));
        //std::cout<<"p :"<<p.x<<" "<<p.y<<"\n";
        //p.y=pic_res-p.y;

        auto p1=p - 1e-3f * n;
        auto p2=p + 1e-3f * n;
        
        p1.y=pic_res-p1.y;
        p2.y=pic_res-p2.y;
        /*
        std::cout<<"n : "<<n.x<<" "<<n.y<<"\n";
        std::cout<<v0.x<<" "<<v0.y<<"\n";
        std::cout<<v1.x<<" "<<v1.y<<"\n";
        std::cout<<"p :"<<p.x<<" "<<p.y<<"\n";
        std::cout<<"p1 :"<<p1.x<<" "<<p1.y<<"\n";
        std::cout<<"p2 :"<<p2.x<<" "<<p2.y<<"\n";
        */
        p1=p1-(pic_res/2)*Vec2f{1,1};
        p2=p2-(pic_res/2)*Vec2f{1,1};
        //std::cout<<"p1 :"<<p1.x<<" "<<p1.y<<"\n";
        //std::cout<<"p2 :"<<p2.x<<" "<<p2.y<<"\n";
        auto dir=normalize(Vec3f{p1.x,p1.y,-f_dist});
        //std::cout<<"dir :"<<dir.x<<" "<<dir.y<<" "<<dir.z<<"\n";
        Rtrc record= rt.intersect(0,0,0, dir.x,dir.y,dir.z);
        auto color_in=record.hit_flag?mesh.colors[record.prim_id]:Vec3f{0,0,0};// = get_color(image, adjoint.height, p - 1e-3f * n);
        //if(record.hit_flag)std::cout<<"hit"<<"\n";
        dir=normalize(Vec3f{p2.x,p2.y,-f_dist});
        record= rt.intersect(0,0,0, dir.x,dir.y,dir.z);
        auto color_out=record.hit_flag?mesh.colors[record.prim_id]:Vec3f{0,0,0};
        // get corresponding adjoint from the adjoint image,
        // multiply with the color difference and divide by the pdf & number of samples.
        auto pdf = pmf / (length(v1 - v0));
        auto weight = Real(1 / (pdf * Real(num_edge_samples)));
        auto adj = dot(color_in - color_out, adjoint.color[yi* adjoint.width + xi]);
        auto d_v0 = Vec2f{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
        auto d_v1 = Vec2f{t * n.x, t * n.y} * adj * weight;
        d_vertices[edge.v0] += d_v0;
        d_vertices[edge.v1] += d_v1;
    }
}

void d_render(ptr<float> shape,
            ptr<float> mv_shape,//get color
            int p_num,
            ptr<unsigned int> indices,
            int tri_num,
            float f_dist,
            int pic_res,
            ptr<float> grad_img,
            ptr<float> normals,//edge sample will use this
            ptr<float> colors,
			ptr<float> d_shape)
{
    TriangleMesh mesh;
    float Mv_shape[3*p_num];//get color
    float *p_mv_shape=mv_shape.get();
    
    for (auto i = 0; i < p_num; i++)
    {
        mesh.vertices.push_back({shape[i], shape[ i + p_num]});
        //std::cout<<mesh.vertices[i].x<<" "<<mesh.vertices[i].y<<"\n";
        Mv_shape[3*i]=p_mv_shape[i];
        Mv_shape[3*i+1]=p_mv_shape[i+p_num];
        Mv_shape[3*i+2]=p_mv_shape[i+2*p_num];
        //std::cout<<Mv_shape[3*i]<<" "<<Mv_shape[3*i+1]<<" "<<Mv_shape[3*i+2]<<"\n";
    }
    
    for (auto i = 0; i < tri_num; i++)
    {
        mesh.indices.push_back({static_cast<int>(indices[6 * i]), static_cast<int>(indices[6 * i + 2]), static_cast<int>(indices[6 * i + 4])});
        mesh.colors.push_back({colors[3*i],colors[3*i+1],colors[3*i+2]});    
    }
    /*
    for(auto i=0;i<mesh.colors.size();i++)
    {
        std::cout<<mesh.colors[i].x<<" "<<mesh.colors[i].y<<" "<<mesh.colors[i].z<<"\n";
    }
    */
    Img adjoint(pic_res, pic_res, Vec3f{1, 1, 1});
    DTriangleMesh d_mesh(mesh.vertices.size());
    auto edges = collect_edges(mesh,normals);
    auto edge_sampler = build_edge_sampler(mesh, edges);

    std::mt19937 rng(1234);
    for (auto i = 0; i < adjoint.width * adjoint.height; i++)
    {
        adjoint.color[i] = Vec3f{grad_img[3 * i],
                                 grad_img[3 * i + 1],
                                 grad_img[3 * i + 2]};
    }
    
    compute_edge_derivatives(mesh, Mv_shape,f_dist,pic_res,edges, edge_sampler, adjoint, 1000 ,
                             rng, d_mesh.vertices);
    
    int s = 0;
    for (auto i : d_mesh.vertices)
    {
        d_shape[s++] = i.x;
        d_shape[s++] = i.y;
        d_shape[s++] = 0;
        d_shape[s++] = 0;
    }
}