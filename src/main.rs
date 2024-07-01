use bmp::{
    consts::{
        ALICE_BLUE, BLACK, BLUE, BLUE_VIOLET, BROWN, GREEN, ORANGE, PINK, RED, WHITE, YELLOW,
    },
    Pixel,
};

// let frame :String = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_050.bmp";
const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_170.bmp";
const COLOR_TO_TRIANGULATE: Pixel = WHITE;

fn main() {
    let img = bmp::open(FRAME).unwrap_or_else(|e| {
        panic!("Failed to open: {}", e);
    });

    let (w, h) = (img.get_width(), img.get_height());
    println!("Loaded bmp, size {} {}", w, h);

    let vertices = detect_edges_as_vertices(&img, COLOR_TO_TRIANGULATE);

    // DEBUG OUTPUT
    let edges_bmp = create_edges_bitmap(&vertices);
    let _ = edges_bmp.save("edges.bmp");

    // TODO Edge walker, find unvisited edges, walks along them to register them as vertex+edge
    // TODO Min size for contours to throw away unwnated domains ??
    // TODO Walk, from bottom to top, try tro walk bottom first agaiin, CW ?
    // TODO Search cycles
    let paths = create_constrained_edges(&vertices);
    // println!("paths {:?}", paths);

    let paths_bmp = create_paths_bitmap(&paths, &vertices.size);
    let _ = paths_bmp.save("paths.bmp");
}

#[derive(Clone, Debug, PartialEq)]
struct Size {
    w: usize,
    h: usize,
}

struct Flags {
    pub verts: Vec<bool>,
    pub size: Size,
}
impl Flags {
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.verts[y * self.size.w + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut bool {
        &mut self.verts[y * self.size.w + x]
    }

    fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.size.w && (y as usize) < self.size.h
    }
}

fn vertex_dist(v1: VertexCoord, v2: VertexCoord) -> usize {
    // v1.0.abs_diff(v2.0) + v1.1.abs_diff(v2.1)
    v1.0.abs_diff(v2.0).max(v1.1.abs_diff(v2.1))
}

pub const MIN_PATH_SIZE: usize = 20;

pub type VertexCoord = (usize, usize);
pub type Path = Vec<VertexCoord>;
pub type Paths = Vec<Path>;

fn register_contours(visited_vertices: &mut Flags, vertices: &Flags) -> Vec<Path> {
    let mut paths = Vec::new();

    for x in (0..vertices.size.w - 3).step_by(3) {
        for y in (0..vertices.size.h - 3).step_by(3) {
            let mut sum_of_verts = 0;
            let mut visited = false;
            let mut side_vertices = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    if vertices.get(x + i, y + j) {
                        sum_of_verts += 1;
                        if i != 1 || j != 1 {
                            side_vertices.push((x + i, y + j));
                        }
                    }
                    if visited_vertices.get(x + i, y + j) {
                        visited = true;
                    }
                }
            }

            // Center is a vertex, there are 3 in total, not visited and dist between the two side vertices is > 1
            let is_path_shape = if vertices.get(x + 1, y + 1) && sum_of_verts == 3 && !visited {
                vertex_dist(side_vertices[0], side_vertices[1]) > 1
            } else {
                false
            };
            if !is_path_shape {
                continue;
            }

            // Exmaple schema:
            // S . .
            // . C .
            // . E .
            let path_center = (x + 1, y + 1);
            let start = side_vertices[0];
            let end = side_vertices[1];

            println!(
                "Found starting path shape x {} y {}, start {:?} end {:?}",
                x, y, start, end
            );

            let path = pathfinding::prelude::astar(
                &start,
                |&(x, y)| {
                    let mut successors = Vec::new();
                    for i in -1..=1 {
                        for j in -1..=1 {
                            if i == 0 && j == 0 {
                                continue;
                            }
                            let (xi, yj) = (x as i32 + i, y as i32 + j);
                            if vertices.is_in_bounds(xi, yj) {
                                let (xi, yj) = (xi as usize, yj as usize);
                                if vertices.get(xi, yj) && path_center != (xi, yj) {
                                    successors.push(((xi, yj), 1));
                                }
                            }
                        }
                    }
                    successors
                },
                |&p| vertex_dist(p, end),
                |&p| p == end,
            );
            // println!("Found path {:?}", path);

            let Some(path) = path else {
                continue;
            };

            println!("Found path with length {}", path.1);

            // if path.1 < MIN_PATH_SIZE {
            //     continue;
            // }

            // Mark pathed vertices as visited
            for v in path.0.iter() {
                *visited_vertices.get_mut(v.0, v.1) = true;
            }

            // TODO Register path as edges and vertices for the triangulation
            paths.push(path.0);
        }
    }

    paths
}

fn create_constrained_edges(vertices: &Flags) -> Paths {
    let mut visited_vertices = Flags {
        verts: vec![false; vertices.size.w * vertices.size.h],
        size: vertices.size.clone(),
    };

    let paths = register_contours(&mut visited_vertices, vertices);
    paths
}

pub const COLORS: &'static [Pixel] = &[
    RED,
    ORANGE,
    BLUE_VIOLET,
    BLUE,
    YELLOW,
    GREEN,
    WHITE,
    PINK,
    ALICE_BLUE,
    BROWN,
];

fn create_paths_bitmap(paths: &Paths, size: &Size) -> bmp::Image {
    let mut paths_bmp = bmp::Image::new(size.w as u32, size.h as u32);

    for (index, path) in paths.iter().enumerate() {
        let color = COLORS[index % COLORS.len()];
        for v in path.iter() {
            paths_bmp.set_pixel(v.0 as u32, v.1 as u32, color);
        }
    }
    paths_bmp
}

fn create_edges_bitmap(vertices: &Flags) -> bmp::Image {
    let mut edges_bmp = bmp::Image::new(vertices.size.w as u32, vertices.size.h as u32);
    for x in 0..vertices.size.w {
        for y in 0..vertices.size.h {
            let color = if vertices.get(x, y) { BLACK } else { WHITE };
            edges_bmp.set_pixel(x as u32, y as u32, color);
        }
    }
    edges_bmp
}

fn search_unvisited_domain_pixel(
    visited_pixels: &Vec<bool>,
    img: &bmp::Image,
    color_to_triangulate: bmp::Pixel,
) -> Option<(u32, u32)> {
    let (w, h) = (img.get_width(), img.get_height());

    let mut non_visited_domain_pixel = None;
    for x in 0..w {
        // Ignore bottom and top rows of the image
        for y in 1..h - 1 {
            if visited_pixels[y as usize * w as usize + x as usize] {
                continue;
            }
            let color = img.get_pixel(x, y);
            if color != color_to_triangulate {
                continue;
            }
            non_visited_domain_pixel = Some((x, y));
            break;
        }
    }
    non_visited_domain_pixel
}

// For a pixel, x and y belong to [0..width-1] of the original bmp image
struct PixelPos {
    x: u32,
    y: u32,
}

fn detect_edges_as_vertices(img: &bmp::Image, color_to_triangulate: bmp::Pixel) -> Flags {
    let (img_w, img_h) = (img.get_width(), img.get_height());
    let mut visited_pixels = vec![false; (img_w * img_h) as usize];

    let vertices_size = Size {
        w: (img_w + 2) as usize,
        h: (img_h + 2) as usize,
    };
    let mut vertices = vec![false; vertices_size.w * vertices_size.h];

    // Search non visited pixel of the color to triangulate
    while let Some(unvisited_domain_pixel) =
        search_unvisited_domain_pixel(&visited_pixels, &img, color_to_triangulate)
    {
        // Initialize the stack
        let mut flood_fill_stack = Vec::new();
        flood_fill_stack.push(PixelPos {
            x: unvisited_domain_pixel.0,
            y: unvisited_domain_pixel.1,
        });

        // Flood fill to find the edges of the color to triangulate
        while let Some(pixel) = flood_fill_stack.pop() {
            visited_pixels[(pixel.y * img_w + pixel.x) as usize] = true;

            for delta in vec![(-1, 0), (0, 1), (1, 0), (0, -1)] {
                let x = pixel.x as i32 + delta.0;
                let y = pixel.y as i32 + delta.1;
                // Ignore bottom and top rows of the image
                if x < 0 || x >= img_w as i32 || y <= 0 || y >= (img_h - 1) as i32 {
                    // Outside of the image, register as an edge vertex
                    vertices[(y + 1) as usize * vertices_size.w + (x + 1) as usize] = true;
                    continue;
                }
                let pos = PixelPos {
                    x: x as u32,
                    y: y as u32,
                };
                let neighbor_color = img.get_pixel(pos.x, pos.y);
                if neighbor_color != color_to_triangulate {
                    // Color change, register as an edge vertex
                    vertices[(y + 1) as usize * vertices_size.w + (x + 1) as usize] = true;
                } else {
                    if !visited_pixels[(pos.y * img_w + pos.x) as usize] {
                        flood_fill_stack.push(pos);
                    }
                }
            }
        }
    }

    Flags {
        verts: vertices,
        size: vertices_size,
    }
}
