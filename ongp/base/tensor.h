#pragma once

#include <torch/torch.h>

namespace ongp
{
    namespace 
    {
    template <class T>
    using Array1d = std::vector<T>;

    template <class T>
    using Array2d = std::vector<std::vector<T>>;

    template <class T>
    using Array3d = std::vector<std::vector<std::vector<T>>>;

    template <class T>
    std::vector<T> Linearize2d(const std::vector<std::vector<T>>& array_2d) {
        std::vector<T> array;
        for (const auto& v : array_2d) {
            for (auto d : v) {
                array.push_back(d);
            }
        }
        return array;
    }


    // Array should use float instead of double. 
    template <class T>
    torch::Tensor Array2dToTensor(const Array2d<T>& array_2d)
    {
        int m = array_2d.size();
        int n = array_2d[0].size();

        // torch::from_blob do not copy original data buffer!
        return torch::from_blob(Linearize2d(array_2d).data(), {m,n}).clone();
    }

    template <class T>
    torch::Tensor Array1dToTensor(const Array1d<T>& array_1d)
    {
        int m = array_1d.size();

   //     std::cout << array_1d[0] << array_1d[1] << array_1d[2] << std::endl;

   //     Array2d<T> array;
   //     Array1d<T> a;
   //     a.push_back(array_1d[0]);
   //     a.push_back(array_1d[1]);
   //     a.push_back(array_1d[2]);
   //     array.push_back(a);

   //     // torch::from_blob do not copy original data buffer!
   //     std::cout << torch::from_blob(const_cast<T*>(array_1d.data()), {m,1}).clone() << std::endl;
   //     std::cout << torch::from_blob(Linearize2d(array).data(), {m,1}).clone() << std::endl;
   //     // from_blob not working???
        return torch::from_blob(const_cast<T*>(array_1d.data()), {m}).clone();
    }

    template <class T>
    torch::Tensor Vector3(const std::initializer_list<T>& list)
    {
        return Array1dToTensor<T>(list);
    }

    inline int random_int(int min, int max) {
        // Returns a random integer in [min,max].
        return static_cast<int>(random_double(min, max+1));
    }

    inline double random_double()
    {
        return rand() / (RAND_MAX+1);
    }

    inline double random_double(double min, double max)
    {
        return min + (max - min) * random_double();
    }

    inline float random_float()
    {
        return rand() / float((RAND_MAX+1.0));
    }

    inline float random_float(float min, float max)
    {
        return min + (max - min) * random_float();
    }

    inline torch::Tensor random_vec3()
    {
        return Array1dToTensor<float>({random_float(), random_float(), random_float()});
    }

    inline torch::Tensor random_vec3(float min, float max)
    {
        return Array1dToTensor<float>({random_float(min, max), random_float(min, max), random_float(min, max)});
    }

    inline torch::Tensor random_in_sphere()
    {
        while(true)
        {
            auto p = random_vec3(-1, 1);
            if (p.norm().item<float>() >= 1) continue;
            return p;
        }
    }

    inline torch::Tensor random_unit_vector()
    {
        auto vec = random_in_sphere();
        return vec / vec.norm();
    }

    inline torch::Tensor random_in_hemisphere(const torch::Tensor& normal) {
        auto in_unit_sphere = random_unit_vector();
        if (torch::dot(in_unit_sphere, normal).item<float>() > 0.0) // In the same hemisphere as the normal
            return in_unit_sphere;
        else
            return -in_unit_sphere;
    }

    inline double clamp(double x, double min, double max) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    inline bool near_zero(const torch::Tensor& vec3)
    {
        const auto s = 1e-6;
        return (fabs(vec3.index({0}).item<float>()) < s) && (fabs(vec3.index({1}).item<float>()) < s) && (fabs(vec3.index({2}).item<float>()) < s);
    }

    inline torch::Tensor reflect(const torch::Tensor& v, const torch::Tensor& n) {
        return v - 2*torch::dot(v,n)*n;
    }

    inline torch::Tensor unit_vector(const torch::Tensor& v) {
        return v / v.norm();
    }

    torch::Tensor refract(const torch::Tensor& uv, const torch::Tensor& n, double etai_over_etat) {
        auto cos_theta = std::fmin(torch::dot(-uv, n).item<float>(), 1.0);
        torch::Tensor r_out_perp =  etai_over_etat * (uv + cos_theta*n);
        torch::Tensor r_out_parallel = -sqrt(fabs(1.0 - (r_out_perp.norm()*r_out_perp.norm()).item<float>())) * n;
        return unit_vector(r_out_perp + r_out_parallel);
    }

    inline bool box_compare(const ObjectSptr a, const ObjectSptr b, int axis) {
        AABB box_a;
        AABB box_b;

        if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
            std::cerr << "No bounding box in bvh_node constructor.\n";

        return box_a.min().e[axis] < box_b.min().e[axis];
    }


    bool box_x_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 0);
    }

    bool box_y_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 1);
    }

    bool box_z_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 2);
    }

    }
}
