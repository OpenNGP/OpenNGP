
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "ongp/base/camera.h"
#include "ongp/base/sphere.h"
#include "ongp/base/scene.h"
#include "ongp/base/tensor.h"
#include "ongp/renderer/render_normal.h"


TEST(RAYTRACING, RENDER) {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // World
    ongp::Scene scene;
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,0,-1}), 0.5));
    scene.Add(std::make_shared<ongp::Sphere>(ongp::Array1dToTensor<float>({0,-100.5,-1}), 100));

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

    ongp::Intrinsics intrs;
    intrs.SetFromFov(60, aspect_ratio, image_height);
    ongp::Camera camera(intrs);
    // I need a camera both for computer vision and graphics
    // Render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto u = double(i) / (image_width-1);
            auto v = double(j) / (image_height-1);
            // extract ray from pixel
            auto r = camera.GenerateRay(u, v);
           // torch::Tensor origin;
           // torch::Tensor direction;
           // ongp::Ray r(origin, direction);
            // shading
            auto pixel_color = ongp::ray_color(r, scene);
            //write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";

}