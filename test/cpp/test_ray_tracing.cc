
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "ongp/base/camera.h"
#include "ongp/base/sphere.h"
#include "ongp/base/scene.h"
#include "ongp/base/tensor.h"
#include "ongp/renderer/lambertian.h"
#include "ongp/renderer/dielectric.h"
#include "ongp/renderer/metal.h"
#include "ongp/renderer/render_normal.h"
#include "ongp/renderer/render_diffuse.h"
#include "ongp/external/delog/delog.h"


//TEST(RAYTRACING, RENDER) {
int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Material
    auto material_ground = std::make_shared<ongp::Lambertian>(ongp::Array1dToTensor<float>({0.8, 0.8, 0.0}));
 //   auto material_center = std::make_shared<ongp::Dielectric>(1.5);
 //   auto material_center = std::make_shared<ongp::Lambertian>(ongp::Array1dToTensor<float>({0.7, 0.3, 0.3}));
 //   auto material_left   = std::make_shared<ongp::Metal>(ongp::Array1dToTensor<float>({0.8, 0.8, 0.8}), 0.3);

    auto material_center = std::make_shared<ongp::Lambertian>(ongp::Array1dToTensor<float>({0.1, 0.2, 0.5}));
    auto material_left   = std::make_shared<ongp::Dielectric>(1.5);
    auto material_right  = std::make_shared<ongp::Metal>(ongp::Array1dToTensor<float>({0.8, 0.6, 0.2}), 0.0);


    // World
    ongp::Scene scene;
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,-100.5,-1}), 100, material_ground));
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,0,1.5}), 0.5, material_center));
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({-1.0,0,1.5}), 0.5, material_left));
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({1.0,0,1.5}), 0.5, material_right));

//    // Camera
//    auto viewport_height = 2.0;
//    auto viewport_width = aspect_ratio * viewport_height;
//    auto focal_length = 1.0;
//
//    auto origin = point3(0, 0, 0);
//    auto horizontal = vec3(viewport_width, 0, 0);
//    auto vertical = vec3(0, viewport_height, 0);
//    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
//
    const int samples_per_pixel = 5;
    const int max_depth = 30;

    ongp::Intrinsics intrs;
    intrs.SetFromFov(3.14/3, aspect_ratio, image_height);
    ongp::Camera camera(intrs);
    // I need a camera both for computer vision and graphics
    // Render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
   // std::cout << "P3\n" << image_width << ' ' << 21 << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
    //for (int j = 21; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
           // auto u = double(i) / (image_width-1);
           // auto v = double(j) / (image_height-1);
           // DELOG(u);
           // DELOG(v);
           auto pixel_color = ongp::Array1dToTensor<float>({0,0,0});
            for (int s = 0; s < samples_per_pixel; ++s) {
                // extract ray from pixel
                auto r = camera.GenerateRay(i+ongp::random_float(), j+ongp::random_float());
            //   std::cout << r.origin() << std::endl;
            //   std::cout << r.direction() << std::endl;
            //   PAUSE();
            // DELOG(u);
            // DELOG(v);
            // torch::Tensor origin;
            // torch::Tensor direction;
            // ongp::Ray r(origin, direction);
                // shading
                pixel_color += ongp::ray_color(r, scene, max_depth);
            }
            ongp::write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";

}

//int main()
//{
//
//    // Image
//    const auto aspect_ratio = 16.0 / 9.0;
//    const int image_width = 400;
//    const int image_height = static_cast<int>(image_width / aspect_ratio);
//
//    // World
//    ongp::Scene scene;
//    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,0,-1}), 0.5));
//    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,-100.5,-1}), 100));
//
////    // Camera
////    auto viewport_height = 2.0;
////    auto viewport_width = aspect_ratio * viewport_height;
////    auto focal_length = 1.0;
////
////    auto origin = point3(0, 0, 0);
////    auto horizontal = vec3(viewport_width, 0, 0);
////    auto vertical = vec3(0, viewport_height, 0);
////    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
////
//
//    ongp::Intrinsics intrs;
//    intrs.SetFromFov(60, aspect_ratio, image_height);
//    ongp::Camera camera(intrs);
//    // I need a camera both for computer vision and graphics
//    // Render
//    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
//
//    for (int j = image_height-1; j >= 0; --j) {
//        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
//        for (int i = 0; i < image_width; ++i) {
//            auto u = double(i) / (image_width-1);
//            auto v = double(j) / (image_height-1);
//            // extract ray from pixel
//            auto r = camera.GenerateRay(u, v);
//           // torch::Tensor origin;
//           // torch::Tensor direction;
//           // ongp::Ray r(origin, direction);
//            // shading
//      //      auto pixel_color = ongp::ray_color(r, scene);
//      //      ongp::write_color(std::cout, pixel_color, 100);
//        }
//    }
//
//    std::cerr << "\nDone.\n";
//
//    return 0;
//}