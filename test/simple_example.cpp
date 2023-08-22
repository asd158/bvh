#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <iostream>

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
int main() {
    StopWatch build;
    build.Start();
    // This is the original data, which may come in some other data type/structure.
    std::vector<Tri> tris;
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
        );
    }
    else {
        std::vector<float> buffPri(1024 * 1024,
                                   0);
        int                validSize = readbyte_from_file("../../test/headm_pri.b",
                                                          reinterpret_cast<char *>(buffPri.data()),
                                                          buffPri.capacity());
        if (validSize > 0) {
            buffPri.resize(validSize / 4);
            std::vector<ushort> buffInd(1024 * 1024,
                                        0);
            validSize = readbyte_from_file("../../test/headm_ind.b",
                                           reinterpret_cast<char *>(buffInd.data()),
                                           buffInd.capacity());
            if (validSize > 0) {
                buffInd.resize(validSize / 2);
                for (int i = 0;
                     i < buffInd.size() / 3;
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

    // Get triangle centers and bounding boxes (required for BVH builder)
    bvh::v2::ThreadPool       thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
    std::vector<BBox>                              bboxes(tris.size());
    std::vector<Vec3>                              centers(tris.size());
    executor.for_each(0,
                      tris.size(),
                      [&](size_t begin,
                          size_t end) {
                          for (size_t i = begin;
                               i < end;
                               ++i) {
                              bboxes[i]  = tris[i].get_bbox();
                              centers[i] = tris[i].get_center();
                          }
                      });
    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::Low;
    auto bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool,
                                                    bboxes,
                                                    centers,
                                                    config);
    build.Stop();
    static constexpr bool       should_permute = true;
    // This precomputes some data to speed up traversal further.
    std::vector<PrecomputedTri> precomputed_tris(tris.size());
    executor.for_each(0,
                      tris.size(),
                      [&](size_t begin,
                          size_t end) {
                          for (size_t i = begin;
                               i < end;
                               ++i) {
                              auto j = should_permute ? bvh.prim_ids[i] : i;
                              precomputed_tris[i] = tris[j];
                          }
                      });
    std::cout << "build time:->" << build.EclipsedMillSecond() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////
    StopWatch runtime;
    runtime.Start();
    for (int i = 0;
         i < 1;
         ++i) {
        auto ray = Ray{
                Vec3(0., 0., 0.), // Ray origin
                Vec3(0., 0., 1), // Ray direction
                0.,               // Minimum intersection distance
                5.              // Maximum intersection distance
        };

        static constexpr size_t invalid_id           = std::numeric_limits<size_t>::max();
        static constexpr size_t stack_size           = 64;
        static constexpr bool   use_robust_traversal = false;

        auto   prim_id = invalid_id;
        Scalar u, v, w;

        // Traverse the BVH and get the u, v coordinates of the closest intersection.
        bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
        bvh.intersect<false, use_robust_traversal>(ray,
                                                   bvh.get_root().index,
                                                   stack,
                                                   [&](size_t begin,
                                                       size_t end) {
                                                       for (size_t i = begin;
                                                            i < end;
                                                            ++i) {
                                                           size_t   j   = should_permute ? i : bvh.prim_ids[i];
                                                           if (auto hit = precomputed_tris[j].intersect(ray)) {
                                                               prim_id = i;
                                                               std::tie(u,
                                                                        v,
                                                                        w) = *hit;
                                                           }
                                                       }
                                                       return prim_id != invalid_id;
                                                   });
        if (prim_id != invalid_id) {
            auto                    p0 = precomputed_tris[prim_id].p0;
            auto                    p1 = precomputed_tris[prim_id].p0 - precomputed_tris[prim_id].e1;
            auto                    p2 = precomputed_tris[prim_id].e2 + precomputed_tris[prim_id].p0;
            bvh::v2::Vec<Scalar, 3> q{}, q_bc{};
            for (int                j  = 0;
                 j < 3;
                 ++j) {
                q[j]    = (u * p1[j] + v * p2[j] + w * p0[j]);
                q_bc[j] = p1[j] + v / (u + v) * (p2[j] - p1[j]);
            }
            auto                    v1 = p0 - q;
            auto                    v2 = p0 - q_bc;
            float                   a0 = dot(v1, v1) / dot(v2, v2);
            v1 = p1 - q_bc;
            v2 = p1 - p2;
            float a1 = dot(v1, v1) / dot(v2, v2);
            std::cout
                    << "Intersection found\n"
                    << "  primitive: " << prim_id << "\n"
                    << "  distance: " << ray.tmax << ", " << ray.tmin << "\n"
                    << "  重心: " << u << ", " << v << ", " << w << std::endl
                    << "  P0   坐标: " << p0[0] << "," << p0[1] << "," << p0[2] << std::endl
                    << "  P1   坐标: " << p1[0] << "," << p1[1] << "," << p1[2] << std::endl
                    << "  P2   坐标: " << p2[0] << "," << p2[1] << "," << p2[2] << std::endl
                    << "  重心q  坐标: " << q[0] << "," << q[1] << "," << q[2] << std::endl
                    << "  延长线交点q_bc: " << q_bc[0] << "," << q_bc[1] << "," << q_bc[2] << std::endl
                    << "  比例系数: " << a0 << ", " << a1 << std::endl;
        }
        else {
            std::cout << "No intersection found" << std::endl;
        }
    }
    runtime.Stop();
    std::cout << "run time:->" << runtime.EclipsedMillSecond() << std::endl;
}
