use bmp::{
    consts::{
        ALICE_BLUE, BLACK, BLUE, BLUE_VIOLET, BROWN, GREEN, ORANGE, PINK, RED, WHITE, YELLOW,
    },
    Pixel,
};

// const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_050.bmp";
// const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_170.bmp";
// const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_180.bmp";
const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_196.bmp";
// const FRAME: &str = "./assets/bad_apple_no_lags_000/bad_apple_no_lags_236.bmp";
const COLOR_TO_TRIANGULATE: Pixel = WHITE;

pub const MIN_PATH_SIZE: usize = 3;

fn main() {
    // Note 0,0 is the upper left corner of the image
    let img = bmp::open(FRAME).unwrap_or_else(|e| {
        panic!("Failed to open: {}", e);
    });

    let pixels_edges = detect_edges(&img, COLOR_TO_TRIANGULATE);
    let output_frame_size = pixels_edges.size;
    // Debug output
    let edges_bmp = create_edges_bitmap(&pixels_edges);
    let _ = edges_bmp.save("edges.bmp");

    let paths = convert_edges_to_paths(&pixels_edges);
    // Debug output
    let paths_bmp = create_paths_bitmap(&paths, output_frame_size);
    let _ = paths_bmp.save("paths.bmp");

    let domains = get_domains_from_paths(output_frame_size, paths);
    // Debug output
    let domains_bmp = create_domains_bitmaps(&domains, output_frame_size);
    for (i, domain_bmp) in domains_bmp.iter().enumerate() {
        let _ = domain_bmp.save(format!("domain_{}.bmp", i));
    }

    let kinds = get_domains_kinds(&domains);
    // Debug output
    let paths_kinds_bmp = create_domains_kinds_bitmap(&domains, kinds, output_frame_size);
    let _ = paths_kinds_bmp.save("paths_kinds.bmp");

    let orientations = get_domains_paths_orientations(&domains);
}

#[derive(Debug, Default, Copy, Clone)]
pub enum Orientation {
    #[default]
    CW,
    CCW,
}
impl Orientation {
    fn opposite(&self) -> Orientation {
        match self {
            Orientation::CW => Orientation::CCW,
            Orientation::CCW => Orientation::CW,
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub enum DomainKind {
    #[default]
    Filled,
    Hollow,
}
impl DomainKind {
    fn opposite(&self) -> DomainKind {
        match self {
            DomainKind::Filled => DomainKind::Hollow,
            DomainKind::Hollow => DomainKind::Filled,
        }
    }
}

pub type DomainId = usize;
#[derive(Default, Clone, Debug)]
pub struct DomainHierarchy {
    children: Vec<DomainId>,
    parent: Option<DomainId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Size {
    w: usize,
    h: usize,
}
impl Size {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h }
    }

    fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h
    }
}

struct Buffer2D<T: Copy> {
    pub data: Vec<T>,
    pub size: Size,
}
impl<T: Copy> Buffer2D<T> {
    pub fn new(value: T, size: Size) -> Self {
        Self {
            data: vec![value; size.w * size.h],
            size,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.size.w + x]
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[y * self.size.w + x]
    }

    fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        self.size.is_in_bounds(x, y)
    }
}

fn vertex_dist(v1: VertexCoord, v2: VertexCoord) -> usize {
    // v1.0.abs_diff(v2.0) + v1.1.abs_diff(v2.1)
    v1.0.abs_diff(v2.0).max(v1.1.abs_diff(v2.1))
}

pub type VertexCoord = (usize, usize);
pub type Path = Vec<VertexCoord>;
pub type Paths = Vec<Path>;

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

fn create_paths_bitmap(paths: &Paths, size: Size) -> bmp::Image {
    let mut paths_bmp = bmp::Image::new(size.w as u32, size.h as u32);

    for (index, path) in paths.iter().enumerate() {
        let color = COLORS[index % COLORS.len()];
        for v in path.iter() {
            paths_bmp.set_pixel(v.0 as u32, v.1 as u32, color);
        }
    }
    paths_bmp
}

fn create_edges_bitmap(pixels_edges: &Buffer2D<bool>) -> bmp::Image {
    let mut edges_bmp = bmp::Image::new(pixels_edges.size.w as u32, pixels_edges.size.h as u32);
    for x in 0..pixels_edges.size.w {
        for y in 0..pixels_edges.size.h {
            let color = if pixels_edges.get(x, y) { BLACK } else { WHITE };
            edges_bmp.set_pixel(x as u32, y as u32, color);
        }
    }
    edges_bmp
}

fn create_domains_bitmaps(domains: &Vec<(Path, Buffer2D<bool>)>, size: Size) -> Vec<bmp::Image> {
    let mut domains_bmps = Vec::new();

    for (index, domain) in domains.iter().enumerate() {
        let mut domain_bmp = bmp::Image::new(size.w as u32, size.h as u32);
        let color = COLORS[index % COLORS.len()];
        for x in 0..domain.1.size.w {
            for y in 0..domain.1.size.h {
                if domain.1.get(x, y) {
                    domain_bmp.set_pixel(x as u32, y as u32, color);
                };
            }
        }
        domains_bmps.push(domain_bmp);
    }
    domains_bmps
}

fn create_domains_kinds_bitmap(
    domains: &Vec<(Path, Buffer2D<bool>)>,
    kinds: Vec<DomainKind>,
    size: Size,
) -> bmp::Image {
    let mut bmp = bmp::Image::new(size.w as u32, size.h as u32);

    for (index, (path, _domain)) in domains.iter().enumerate() {
        let color = match kinds[index] {
            DomainKind::Filled => YELLOW,
            DomainKind::Hollow => BLUE_VIOLET,
        };
        for v in path.iter() {
            bmp.set_pixel(v.0 as u32, v.1 as u32, color);
        }
    }
    bmp
}

fn search_unvisited_domain_pixel(
    visited_pixels: &Buffer2D<bool>,
    img: &bmp::Image,
    color_to_triangulate: bmp::Pixel,
) -> Option<(u32, u32)> {
    let mut non_visited_domain_pixel = None;
    // We can safely borders of the image if we want here. Should not have any influence, just helps to avoid malformed images.
    for x in 1..img.get_width() - 1 {
        for y in 1..img.get_height() - 1 {
            if visited_pixels.get(x as usize, y as usize) {
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

const DIRECT_NEIGHBORS: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

fn detect_edges(img: &bmp::Image, color_to_triangulate: bmp::Pixel) -> Buffer2D<bool> {
    let img_size = Size::new(img.get_width() as usize, img.get_height() as usize);
    let mut visited_pixels = Buffer2D::new(false, img_size);

    // We increase size by 1 to be able to ensure void borders in edges_buffer
    let buffer_size = Size::new(img_size.w + 2, img_size.h + 2);
    let mut edges_buffer = Buffer2D::new(false, buffer_size);

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
            *visited_pixels.get_mut(pixel.x as usize, pixel.y as usize) = true;

            for delta in DIRECT_NEIGHBORS.iter() {
                let (x, y) = (pixel.x as i32 + delta.0, pixel.y as i32 + delta.1);
                if !img_size.is_in_bounds(x, y) {
                    // Outside of the image, register as an edge vertex
                    // Use pixel as the position, to leave void borders in edges_buffer
                    *edges_buffer.get_mut((pixel.x + 1) as usize, (pixel.y + 1) as usize) = true;
                    continue;
                }
                let pos = PixelPos {
                    x: x as u32,
                    y: y as u32,
                };
                let neighbor_color = img.get_pixel(pos.x, pos.y);
                if neighbor_color != color_to_triangulate {
                    // Color change, register as an edge vertex
                    *edges_buffer.get_mut((x + 1) as usize, (y + 1) as usize) = true;
                } else {
                    if !visited_pixels.get(pos.x as usize, pos.y as usize) {
                        flood_fill_stack.push(pos);
                    }
                }
            }
        }
    }

    edges_buffer
}

fn convert_edges_to_paths(pixels_edges: &Buffer2D<bool>) -> Vec<Path> {
    let mut visited_vertices = Buffer2D::new(false, pixels_edges.size);
    let mut paths = Vec::new();

    // Iterate the whole image with 3x3 squares
    for x in (0..pixels_edges.size.w - 3).step_by(1) {
        for y in (0..pixels_edges.size.h - 3).step_by(1) {
            // Find valid path-like shapes
            let mut visited = false;
            let mut side_vertices = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    if pixels_edges.get(x + i, y + j) {
                        if i != 1 || j != 1 {
                            side_vertices.push((x + i, y + j));
                        }
                    }
                    if visited_vertices.get(x + i, y + j) {
                        visited = true;
                    }
                }
            }

            // Center is a vertex, not visited, there are exactly 2 sides vertices, and the dist between the two side vertices is > 1
            let is_valid_path_shape = pixels_edges.get(x + 1, y + 1)
                && !visited
                && side_vertices.len() == 2
                && vertex_dist(side_vertices[0], side_vertices[1]) > 1;
            if !is_valid_path_shape {
                continue;
            }

            // Example path-like shape:
            // S . .
            // . C .
            // . E .
            let path_center = (x + 1, y + 1);
            let path_start = side_vertices[0];
            let path_end = side_vertices[1];

            // Pathfind from Start to End, ignoring Center
            let path = pathfinding::prelude::astar(
                &path_start,
                |&(x, y)| {
                    let mut successors = Vec::new();
                    for i in -1..=1 {
                        for j in -1..=1 {
                            if i == 0 && j == 0 {
                                continue;
                            }
                            let (xi, yj) = (x as i32 + i, y as i32 + j);
                            if pixels_edges.is_in_bounds(xi, yj)
                                && !visited_vertices.get(xi as usize, yj as usize)
                            {
                                let (xi, yj) = (xi as usize, yj as usize);
                                if pixels_edges.get(xi, yj) && path_center != (xi, yj) {
                                    successors.push(((xi, yj), 1));
                                }
                            }
                        }
                    }
                    successors
                },
                |&p| vertex_dist(p, path_end),
                |&p| p == path_end,
            );

            let Some(mut path) = path else {
                continue;
            };

            // Re-add the center vertex to the path
            path.0.push(path_center);

            println!("Found path with length {}", path.1);

            if path.1 + 1 < MIN_PATH_SIZE {
                continue;
            }

            // Mark pathed vertices as visited
            for v in path.0.iter() {
                *visited_vertices.get_mut(v.0, v.1) = true;
            }

            paths.push(path.0);
        }
    }

    paths
}

fn get_domains_from_paths(frame_size: Size, mut paths: Paths) -> Vec<(Path, Buffer2D<bool>)> {
    let mut domains = Vec::new();

    while let Some(path) = paths.pop() {
        // Initialize an emtpy domain
        let mut domain = Buffer2D::new(false, frame_size);
        // Mark paths vertices as in the domain
        for p in path.iter() {
            *domain.get_mut(p.0, p.1) = true;
        }

        // Initialize the stack
        let mut flood_fill_stack = Vec::new();
        flood_fill_stack.push(PixelPos { x: 0, y: 0 });

        // Flood fill the domain exterior
        while let Some(pixel) = flood_fill_stack.pop() {
            *domain.get_mut(pixel.x as usize, pixel.y as usize) = true;

            for delta in DIRECT_NEIGHBORS.iter() {
                let (x, y) = (pixel.x as i32 + delta.0, pixel.y as i32 + delta.1);
                if !domain.is_in_bounds(x, y) {
                    // Ignore OOB points
                    continue;
                }
                let pos = PixelPos {
                    x: x as u32,
                    y: y as u32,
                };
                if !domain.get(pos.x as usize, pos.y as usize) {
                    flood_fill_stack.push(pos);
                }
            }
        }

        // Invert the domain data to fit inside the path
        for v in domain.data.iter_mut() {
            *v = !*v;
        }

        domains.push((path, domain));
    }

    domains
}

fn get_domains_kinds(domains: &Vec<(Path, Buffer2D<bool>)>) -> Vec<DomainKind> {
    // Find paths hierarchy (who contains who), then invert orientation for each level
    let mut hierarchies = vec![DomainHierarchy::default(); domains.len()];
    for (i, (path, domain)) in domains.iter().enumerate() {
        for (j, (other_path, other_domain)) in domains[i + 1..domains.len()].iter().enumerate() {
            let j = j + i + 1;

            if domain.get(other_path[0].0, other_path[0].1)
                && !other_domain.get(path[0].0, path[0].1)
            {
                // i C j
                hierarchies[i].children.push(j);
                hierarchies[j].parent = Some(i);
            } else if other_domain.get(path[0].0, path[0].1)
                && !domain.get(other_path[0].0, other_path[0].1)
            {
                // j C i
                hierarchies[j].children.push(i);
                hierarchies[i].parent = Some(j);
            }
        }
    }

    let mut kinds = vec![DomainKind::default(); domains.len()];
    // Search all the root domains (not contained by any other domains) and set orientaitons for their hierarchy
    for (id, hierarchy) in hierarchies.iter().enumerate() {
        if hierarchy.parent.is_none() {
            set_kinds_with_recursive_hierarchy_walk(id, kinds[id], &mut kinds, &hierarchies);
        }
    }

    kinds
}

pub fn set_kinds_with_recursive_hierarchy_walk(
    parent_hierarchy_id: DomainId,
    parent_kind: DomainKind,
    kinds: &mut Vec<DomainKind>,
    hierarchies: &Vec<DomainHierarchy>,
) {
    let child_kind = parent_kind.opposite();
    for child_id in hierarchies[parent_hierarchy_id].children.iter() {
        kinds[*child_id] = child_kind;
        set_kinds_with_recursive_hierarchy_walk(*child_id, child_kind, kinds, hierarchies);
    }
}

fn get_domains_paths_orientations(domains: &Vec<(Path, Buffer2D<bool>)>) -> Vec<Orientation> {
    // Paths orientations are != based on the starting shape+location.. See if it lies to the right or left of the path => gives the path orientation.
    let mut orientations = Vec::with_capacity(domains.len());
    for (path, domain) in domains.iter() {
        let delta_x = (path[1].0 as i32) - (path[0].0 as i32);
        let delta_y = (path[1].1 as i32) - (path[0].1 as i32);
        // Quick & dirty. Might discard left neighbor
        let (right_neighbor, _left_neigh) = if delta_x == 1 && delta_y == -1 {
            (
                (path[0].0 as i32 + 1, path[0].1 as i32),
                (path[0].0 as i32, path[0].1 as i32 - 1),
            )
        } else if delta_x == 1 && delta_y == 1 {
            (
                (path[0].0 as i32, path[0].1 as i32 + 1),
                (path[0].0 as i32 + 1, path[0].1 as i32),
            )
        } else if delta_x == -1 && delta_y == -1 {
            (
                (path[0].0 as i32, path[0].1 as i32 - 1),
                (path[0].0 as i32 - 1, path[0].1 as i32),
            )
        } else if delta_x == -1 && delta_y == 1 {
            (
                (path[0].0 as i32 - 1, path[0].1 as i32),
                (path[0].0 as i32, path[0].1 as i32 + 1),
            )
        } else {
            // delta_x == 0 || delta_y == 0
            (
                (path[0].0 as i32 - delta_y, path[0].1 as i32 + delta_x),
                (path[0].0 as i32 + delta_y, path[0].1 as i32 - delta_x),
            )
        };

        let path_vertices_orientation =
            if domain.get(right_neighbor.0 as usize, right_neighbor.1 as usize) {
                Orientation::CW
            } else {
                Orientation::CCW
            };
        orientations.push(path_vertices_orientation);
    }

    orientations
}
