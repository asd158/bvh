#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <iostream>
#include <vector>

static int readbyte_from_file(const std::string &path,
                              char *inputBuff,
                              int inputBuffSize) {
    if (path.empty() || inputBuff == nullptr || inputBuffSize <= 0) {
        return -1;
    }
    FILE *readFile = nullptr;
    readFile = fopen(path.c_str(),
                     "rb");
    if (readFile == nullptr) {
        return -2;
    }
    if (ferror(readFile)) {
        return -3;
    }
    int    total = 0;
    size_t len;
    while (!feof(readFile)) {
        len = static_cast<int>(fread(inputBuff + total,
                                     1,
                                     1024,
                                     readFile));
        total += len;
    }
    fclose(readFile);
    return total;
}

#include <chrono>

class StopWatch {
public:
    void Start();
    void Stop();
    [[nodiscard]] long long EclipsedMillSecond() const;
private:
    bool                                  _isStart{false};
    long long                             _finalFS{-1};
    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _end;
};

void StopWatch::Start() {
    if (_isStart) {
        return;
    }
    _isStart = true;
    _start   = std::chrono::steady_clock::now();
}

void StopWatch::Stop() {
    if (!_isStart) {
        return;
    }
    _isStart = false;
    _end     = std::chrono::steady_clock::now();
    _finalFS = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
}

long long StopWatch::EclipsedMillSecond() const {
    if (!_isStart) {
        return _finalFS;
    }
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - _start).count();
}
using Scalar = float;
using Vec3 = bvh::v2::Vec<Scalar, 3>;
using BBox = bvh::v2::BBox<Scalar, 3>;
using Tri = bvh::v2::Tri<Scalar, 3>;
using Node = bvh::v2::Node<Scalar, 3>;
using Bvh = bvh::v2::Bvh<Node>;
using Ray = bvh::v2::Ray<Scalar, 3>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <iostream>

void build_dat( const std::string & tempPath,const std::vector<Tri> &tris) {
    std::vector<BBox>                              bboxes(tris.size());
    std::vector<Vec3>                              centers(tris.size());
    for (size_t                                    i = 0;
         i < tris.size();
         ++i) {
        bboxes[i]  = tris[i].get_bbox();
        centers[i] = tris[i].get_center();
    }
    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Low;
    auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes,
                                                    centers,
                                                    config);
    std::ofstream out(tempPath, std::ofstream::binary);
    bvh::v2::StdOutputStream stream(out);
    bvh.serialize(stream);
}

void make_result(Bvh &bvh,
                 std::vector<PrecomputedTri> &precomputed_tris,
                 const Vec3 &vert,
                 float result[4]) {
    //result[0]: p0到重心点的比例
    //result[1]: p1到bc上交点的比例
    //result[2]: orig到重心点的比例
    Vec3 orig{0, 0, 0};
    auto disVect = orig - vert;

    auto                              ray        = Ray{
            orig, -disVect, 0., 10
    };
    static constexpr size_t           invalid_id = std::numeric_limits<size_t>::max();
    auto                              prim_id    = invalid_id;
    Scalar                            u, v, w;
    // Traverse the BVH and get the u, v coordinates of the closest intersection.
    bvh::v2::GrowingStack<Bvh::Index> stack;
    bvh.intersect<false, true>(ray,
                               bvh.get_root().index,
                               stack,
                               [&](size_t begin,
                                   size_t end) {
                                   for (size_t i = begin;
                                        i < end;
                                        ++i) {
                                       if (auto hit = precomputed_tris[i].intersect(ray)) {
                                           prim_id = i;
                                           std::tie(u, v, w) = *hit;
                                       }
                                   }
                                   return prim_id != invalid_id;
                               });
    if (prim_id != invalid_id) {
        auto                    p0 = precomputed_tris[prim_id].p0;
        auto                    p1 = precomputed_tris[prim_id].p0 - precomputed_tris[prim_id].e1;
        auto                    p2 = precomputed_tris[prim_id].e2 + precomputed_tris[prim_id].p0;
        bvh::v2::Vec<Scalar, 3> q{}, q_bc{};
        //赋值result
        result[2] = ray.tmax / sqrt(dot(disVect, disVect));
        memcpy(&result[3], &prim_id, 4);
        if (std::abs(u + v) <= 0.000001f) {
            result[0] = 0;
            result[1] = 0;
        }
        else {
            for (int j  = 0;
                 j < 3;
                 ++j) {
                q[j]    = (u * p1[j] + v * p2[j] + w * p0[j]);
                q_bc[j] = p1[j] + v / (u + v) * (p2[j] - p1[j]);
            }
            auto     v1 = p0 - q;
            auto     v2 = p0 - q_bc;
            auto     r1 = sqrt(dot(v1, v1) / dot(v2, v2));
            v1 = p1 - q_bc;
            v2 = p1 - p2;
            auto r2 = sqrt(dot(v1, v1) / dot(v2, v2));
            if (std::abs(r1) <= 0.000001f) {
                r1 = 0;
            }
            if (std::abs(r2) <= 0.000001f) {
                r2 = 0;
            }
            result[0] = r1;
            result[1] = r2;
//            std::cout
//                    << "  primitive: " << prim_id << "\n"
//                    << "  distance: " << ray.tmax << ", " << ray.tmin << "\n"
//                    << "  重心: " << u << ", " << v << ", " << w << std::endl
//                    << "  P1   坐标: " << p1[0] << "," << p1[1] << "," << p1[2] << std::endl
//                    << "  P2   坐标: " << p2[0] << "," << p2[1] << "," << p2[2] << std::endl
//                    << "  重心q  坐标: " << q[0] << "," << q[1] << "," << q[2] << std::endl
//                    << "  延长线交点q_bc: " << q_bc[0] << "," << q_bc[1] << "," << q_bc[2] << std::endl
//                    << "  比例系数: " << result[0] << ", " << result[1] << ", " << result[2] << std::endl;
        }
    }
    else {
        result[0] = -1;
        result[1] = -1;
        result[2] = 0;
        int idx = -1;
        memcpy(&result[3], &idx, 4);
    }
}

int main() {
    StopWatch build;
    build.Start();
    // This is the original data, which may come in some other data type/structure.
    std::vector<Tri>                               tris;
    if (false) {
        tris.emplace_back(
                //p0点重复
                Vec3(0.0,
                     0.0,
                     1.0),
                Vec3(1.0,
                     1.0,
                     1.0),
                Vec3(-1.0,
                     1.0,
                     1.0)

//                //p1点重复
//                Vec3(1.0,
//                     -1.0,
//                     1.0),
//                Vec3(0.0,
//                     0.0,
//                     1.0),
//                Vec3(-1.0,
//                     -1.0,
//                     1.0)

//                //p2点重复
//                Vec3(1.0,
//                     -1.0,
//                     1.0),
//                Vec3(1.0,
//                     1.0,
//                     1.0),
//                Vec3(0.0,
//                     0.0,
//                     1.0)


//                //p0-p1边上
//                Vec3(-2.0,
//                     -2.0,
//                     1.0),
//                Vec3(1.0,
//                     1.0,
//                     1.0),
//                Vec3(-1.0,
//                     1.0,
//                     1.0)
//
//                //p0 -p2边上
//                Vec3(1.0,
//                     -1.0,
//                     1.0),
//                Vec3(1.0,
//                     1.0,
//                     1.0),
//                Vec3(-1.0,
//                     1.0,
//                     1.0)
        );
    }
    else {
        std::vector<float> buffPri(1024 * 1024);
        int                validSize = readbyte_from_file("../../test/headm_pri.b",
                                                          reinterpret_cast<char *>(buffPri.data()),
                                                          buffPri.capacity());
        if (validSize > 0) {
            buffPri.resize(validSize / 4);
            std::vector<unsigned short> buffInd(1024 * 1024);
            validSize = readbyte_from_file("../../test/headm_ind.b",
                                           reinterpret_cast<char *>(buffInd.data()),
                                           buffInd.capacity());
            if (validSize > 0) {
                buffInd.resize(validSize / 2);
                for (int i = 0;
                     i < (int) buffInd.size() / 3;
                     ++i) {
                    int idx[3] = {buffInd[i * 3 + 0], buffInd[i * 3 + 1], buffInd[i * 3 + 2]};
                    tris.emplace_back(
                            Vec3(buffPri[idx[0] * 3],
                                 buffPri[idx[0] * 3 + 1],
                                 buffPri[idx[0] * 3 + 2]),
                            Vec3(buffPri[idx[1] * 3],
                                 buffPri[idx[1] * 3 + 1],
                                 buffPri[idx[1] * 3 + 2]),
                            Vec3(buffPri[idx[2] * 3],
                                 buffPri[idx[2] * 3 + 1],
                                 buffPri[idx[2] * 3 + 2])
                    );
                }
            }
            else {
                abort();
            }
        }
        else {
            abort();
        }
    }
    std::vector<BBox>                              bboxes(tris.size());
    std::vector<Vec3>                              centers(tris.size());
    for (size_t                                    i = 0;
         i < tris.size();
         ++i) {
        bboxes[i]  = tris[i].get_bbox();
        centers[i] = tris[i].get_center();
    }
    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Low;
    auto                        bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes,
                                                                           centers,
                                                                           config);
    // This precomputes some data to speed up traversal further.
    std::vector<PrecomputedTri> precomputed_tris(tris.size());
    for (size_t                 i   = 0;
         i < tris.size();
         ++i) {
        auto j = bvh.prim_ids[i];
        precomputed_tris[i] = tris[j];
    }
    build.Stop();
    std::cout << "build time:->" << build.EclipsedMillSecond() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    float result[4];
    Vec3  vert(0., 0., 1);
    {
        StopWatch calcT;
        calcT.Start();
        std::cout << "-------------计算-----------" << std::endl;
        for (int i = 0;
             i < 200000;
             ++i) {
            auto a = vert + Vec3(0, 0, 0.0001f);
            make_result(bvh, precomputed_tris, a, result);
        }
        calcT.Stop();
        std::cout << "-计算:->" << calcT.EclipsedMillSecond() << std::endl;
    }
    {
        StopWatch revCalcT;
        revCalcT.Start();
        std::cout << "-------------反计算-----------" << std::endl;
        int primIdx = -1;
        memcpy(&primIdx, &result[3], 4);
        if (primIdx >= 0) {
            std::cout << "距离比例: " << result[2] << std::endl;
            auto &tri = precomputed_tris[primIdx];
            if ((result[1] == 0) && (result[0] == 0)) {
                //就是p0
                std::cout << "-交点坐标1: " << tri.p0[0] << "," << tri.p0[1] << "," << tri.p0[2] << std::endl;
            }
            else {
                Vec3     p{};
                for (int i = 0;
                     i < 200000;
                     ++i) {
                    auto p_bc = result[1] * (tri.e2 + tri.e1) + (tri.p0 - tri.e1);
                    p = result[0] * (p_bc - tri.p0) + tri.p0;
                }
                std::cout << "-交点坐标2: " << p[0] << "," << p[1] << "," << p[2] << std::endl;
            }
        }
        else {
            std::cout << "-没有交点 " << std::endl;
        }
        revCalcT.Stop();
        std::cout << "-反计算:->" << revCalcT.EclipsedMillSecond() << std::endl;
    }
}
