use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufWriter, Write},
    ops::RangeInclusive,
};

use bmp::{
    consts::{
        ALICE_BLUE, BLACK, BLUE, BLUE_VIOLET, BROWN, GREEN, ORANGE, PINK, RED, WHITE, YELLOW,
    },
    Pixel,
};
use serde::Serialize;

#[cfg(feature = "profile_traces")]
use tracing::{span, Level};
#[cfg(feature = "profile_traces")]
use tracing_subscriber::{layer::SubscriberExt, Registry};
#[cfg(feature = "profile_traces")]
use tracing_tracy::TracyLayer;

// Quick & dirty, could be a feature
pub const DEBUG_OUTPUT: bool = false;
// pub const DEBUG_OUTPUT: bool = true;
// pub const FRAMES_TO_PROCESS_RANGE: RangeInclusive<usize> = 0..=6472;
pub const FRAMES_TO_PROCESS_RANGE: RangeInclusive<usize> = 86..=145;
// pub const FRAMES_TO_PROCESS_RANGE: RangeInclusive<usize> = 145..=145;
pub const RENAME_OUTPUT_INDEX_START: usize = 0;

pub const COLOR_TO_TRIANGULATE: Pixel = BLACK;

pub const DEFAULT_PIXEL_GREYSCALE_BLACK_THRESHOLD: f32 = 0.45;
pub const DEFAULT_MIN_PATH_SIZE: usize = 4;

/// Quick & easy way to tweak some frames where small paths cause issues
fn frame_min_path_length(frame_index: usize) -> usize {
    match frame_index {
        // Sometimes small loops can disable a bigger loop by "stealing" a part of the big loop.
        // 66 => 8,
        _ => DEFAULT_MIN_PATH_SIZE,
    }
}

#[derive(Debug, Serialize)]
pub struct Frame {
    pub vertices: Vec<(i32, i32)>,
    pub edges: Vec<(usize, usize)>,
}
impl Frame {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
        }
    }
}

fn main() {
    #[cfg(feature = "profile_traces")]
    {
        let subscriber = Registry::default().with(TracyLayer::default());
        tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
    }
    process_all_bmp_files();
    // _rename_some_files();
}

fn _rename_some_files() {
    let mut output_index = RENAME_OUTPUT_INDEX_START;
    for i in FRAMES_TO_PROCESS_RANGE {
        let from = format!("./assets/input_frames/{:05}.bmp", i);
        let to = format!("./assets/input_frames/{:05}.bmp", output_index);
        fs::rename(from, to).unwrap();
        output_index += 1;
    }
}

fn process_all_bmp_files() {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "process_all_bmp_files").entered();

    // TODO Black & white frame: may invert the color to triangulate
    let mut frames = Vec::new();
    for i in FRAMES_TO_PROCESS_RANGE {
        println!("{}", format!("./assets/input_frames_hq/{:05}.bmp", i));
        let frame_bmp_img = bmp::open(format!("./assets/input_frames_hq/{:05}.bmp", i)).unwrap();
        let processed_frame = process_frame(&frame_bmp_img, i);
        frames.push(processed_frame);
    }

    let file = File::create("bad_apple_frames.msgpack").unwrap();
    let mut writer = BufWriter::new(file);
    let bytes = rmp_serde::to_vec(&frames).unwrap();
    writer.write_all(&bytes).unwrap();
    writer.flush().unwrap();
}

fn process_frame(img: &bmp::Image, frame_index: usize) -> Frame {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "process_frame").entered();

    let monochrome_bmp = convert_to_monochrome(img);
    if DEBUG_OUTPUT {
        let _ = monochrome_bmp.save("1_monochrome.bmp");
    }

    // Note 0,0 is the upper left corner of the image
    let pixels_edges = detect_edges(&monochrome_bmp, COLOR_TO_TRIANGULATE);
    let output_frame_size = pixels_edges.size;
    if DEBUG_OUTPUT {
        let edges_bmp = create_edges_bitmap(&pixels_edges);
        let _ = edges_bmp.save("2_edges.bmp");
    }

    let paths = convert_edges_to_paths(&pixels_edges, frame_index);
    if DEBUG_OUTPUT {
        let paths_bmp = create_paths_bitmap(&paths, output_frame_size);
        let _ = paths_bmp.save("3_paths.bmp");
    }

    let mut domains = get_domains_from_paths(paths);
    if DEBUG_OUTPUT {
        let domains_bmp = create_domains_bitmaps(&domains, output_frame_size);
        for (i, domain_bmp) in domains_bmp.iter().enumerate() {
            let _ = domain_bmp.save(format!("4_{}_domain.bmp", i));
        }
    }

    let kinds = get_domains_kinds(&domains);
    if DEBUG_OUTPUT {
        let domains_kinds_bmp = create_domains_kinds_bitmap(&domains, &kinds, output_frame_size);
        let _ = domains_kinds_bmp.save("5_domains_kinds.bmp");
    }

    let orientations = get_domains_paths_orientations(&domains);

    // Paths simplification. Remove (some) redondant vertices. Could do more
    simplify_paths_vertices(&mut domains);
    if DEBUG_OUTPUT {
        let simplified_paths_bmp = create_domains_kinds_bitmap(&domains, &kinds, output_frame_size);
        let _ = simplified_paths_bmp.save("6_simplified_paths.bmp");
    }

    convert_domains_to_frame(&domains, &kinds, &orientations)
}

fn convert_domains_to_frame(
    domains: &Vec<Domain>,
    kinds: &Vec<DomainKind>,
    orientations: &Vec<Orientation>,
) -> Frame {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "convert_domains_to_frame").entered();

    let mut frame = Frame::new();
    // Convert path to vertices
    for ((domain, kind), orientation) in domains.iter().zip(kinds).zip(orientations) {
        let first_vertex = frame.vertices.len();
        for v in domain.path.pos.iter() {
            // TODO Re-Center
            // Y bitmap axis is inverted.
            frame.vertices.push((v.0 as i32, -(v.1 as i32)));
        }
        let last_vertex = frame.vertices.len() - 1;

        let invert = match orientation {
            Orientation::CW => match kind {
                DomainKind::Filled => false,
                DomainKind::Hollow => true,
            },
            Orientation::CCW => match kind {
                DomainKind::Filled => true,
                DomainKind::Hollow => false,
            },
        };
        match invert {
            true => {
                frame
                    .edges
                    .extend((first_vertex..last_vertex).map(|i| (i + 1, i)));
                frame.edges.push((first_vertex, last_vertex));
            }
            false => {
                frame
                    .edges
                    .extend((first_vertex..last_vertex).map(|i| (i, i + 1)));
                frame.edges.push((last_vertex, first_vertex));
            }
        }
    }
    frame
}

fn convert_to_monochrome(img: &bmp::Image) -> bmp::Image {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "convert_to_monochrome").entered();

    let mut monochrome_bmp = bmp::Image::new(img.get_width() - 2, img.get_height() - 2);
    // Ignore the borders, some have bad data in the input images

    // Y first for cache efficiency
    for y in 1..img.get_height() - 1 {
        for x in 1..img.get_width() - 1 {
            let color = img.get_pixel(x, y);
            let greyscale = (color.r as f32 + color.g as f32 + color.b as f32) / (3. * 255.);
            if greyscale > DEFAULT_PIXEL_GREYSCALE_BLACK_THRESHOLD {
                monochrome_bmp.set_pixel(x - 1, y - 1, BLACK)
            } else {
                monochrome_bmp.set_pixel(x - 1, y - 1, WHITE)
            }
        }
    }
    monochrome_bmp
}

fn simplify_paths_vertices(domains: &mut Vec<Domain>) {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "simplify_paths_vertices").entered();

    for domain in domains.iter_mut() {
        let raw_path = &mut domain.path.pos;
        let mut simplified_path = RawPath::new();

        let mut current_delta = (
            (raw_path[0].0 as i32) - (raw_path[raw_path.len() - 1].0 as i32),
            (raw_path[0].1 as i32) - (raw_path[raw_path.len() - 1].1 as i32),
        );
        for i in 1..raw_path.len() {
            let delta = (
                (raw_path[i].0 as i32) - (raw_path[i - 1].0 as i32),
                (raw_path[i].1 as i32) - (raw_path[i - 1].1 as i32),
            );
            if delta != current_delta {
                simplified_path.push(raw_path[i - 1]);
            }
            current_delta = delta;
        }
        // Quick & dirty: simply force the end point to be in
        simplified_path.push(raw_path[raw_path.len() - 1]);
        *raw_path = simplified_path;
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub enum Orientation {
    #[default]
    CW,
    CCW,
}
impl Orientation {
    pub fn opposite(&self) -> Orientation {
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

    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h
    }
}

#[derive(Clone)]
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

    fn is_valid(&self, x: i32, y: i32) -> bool {
        self.size.is_valid(x, y)
    }
}

pub type VertexCoord = (usize, usize);

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
        for v in path.pos.iter() {
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

fn _create_multidomains_bitmap(multidomains: &Buffer2D<Option<(usize, usize)>>) -> bmp::Image {
    let mut domains_bmp = bmp::Image::new(multidomains.size.w as u32, multidomains.size.h as u32);
    for x in 0..multidomains.size.w {
        for y in 0..multidomains.size.h {
            match multidomains.get(x, y) {
                Some((id, _)) => {
                    domains_bmp.set_pixel(x as u32, y as u32, COLORS[id % COLORS.len()])
                }
                None => domains_bmp.set_pixel(x as u32, y as u32, BLACK),
            }
        }
    }
    domains_bmp
}

fn create_domains_bitmaps(domains: &Vec<Domain>, size: Size) -> Vec<bmp::Image> {
    let mut domains_bmps = Vec::new();

    for (index, domain) in domains.iter().enumerate() {
        let mut domain_bmp = bmp::Image::new(size.w as u32, size.h as u32);
        let color = COLORS[index % COLORS.len()];
        for x in 0..size.w {
            for y in 0..size.h {
                if domain.get_global(x, y) {
                    domain_bmp.set_pixel(x as u32, y as u32, color);
                };
            }
        }
        domains_bmps.push(domain_bmp);
    }
    domains_bmps
}

fn create_domains_kinds_bitmap(
    domains: &Vec<Domain>,
    kinds: &Vec<DomainKind>,
    size: Size,
) -> bmp::Image {
    let mut bmp = bmp::Image::new(size.w as u32, size.h as u32);

    for (index, domain) in domains.iter().enumerate() {
        let color = match kinds[index] {
            DomainKind::Filled => YELLOW,
            DomainKind::Hollow => BLUE_VIOLET,
        };
        for v in domain.path.pos.iter() {
            bmp.set_pixel(v.0 as u32, v.1 as u32, color);
        }
    }
    bmp
}

fn search_unvisited_domain_pixel(
    visited_pixels: &Buffer2D<bool>,
    img: &bmp::Image,
    color_to_triangulate: bmp::Pixel,
    from_x: u32,
) -> Option<(u32, u32)> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "search_unvisited_domain_pixel").entered();

    // We can safely ignore borders of the image if we want here. Should not have any influence, just helps to avoid malformed images.
    for x in from_x..img.get_width() - 1 {
        for y in 1..img.get_height() - 1 {
            if visited_pixels.get(x as usize, y as usize) {
                continue;
            }
            let color = img.get_pixel(x, y);
            if color != color_to_triangulate {
                continue;
            }
            return Some((x, y));
        }
    }
    None
}

// For a pixel, x and y belong to [0..width-1] of the original bmp image
struct PixelPos {
    x: u32,
    y: u32,
}
impl PixelPos {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

const DIRECT_NEIGHBORS: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
const ALL_NEIGHBORS: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
];

fn detect_edges(img: &bmp::Image, color_to_triangulate: bmp::Pixel) -> Buffer2D<bool> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "detect_edges").entered();

    let img_size = Size::new(img.get_width() as usize, img.get_height() as usize);
    let mut visited_pixels = Buffer2D::new(false, img_size);

    // We increase size by 1 to be able to ensure void borders in edges_buffer
    let buffer_size = Size::new(img_size.w + 2, img_size.h + 2);
    let mut edges_buffer = Buffer2D::new(false, buffer_size);

    // Search non visited pixel of the color to triangulate
    let mut flood_fill_stack = Vec::new();
    // Remember where we were in the search, as a quick optimization
    let mut from_x = 1;
    while let Some(unvisited_domain_pixel) =
        search_unvisited_domain_pixel(&visited_pixels, &img, color_to_triangulate, from_x)
    {
        // Update search base index
        from_x = unvisited_domain_pixel.0;
        // Initialize the stack
        flood_fill_stack.push(PixelPos {
            x: unvisited_domain_pixel.0,
            y: unvisited_domain_pixel.1,
        });

        // Flood fill to find the edges of the color to triangulate
        while let Some(pixel) = flood_fill_stack.pop() {
            *visited_pixels.get_mut(pixel.x as usize, pixel.y as usize) = true;

            for delta in DIRECT_NEIGHBORS.iter() {
                let (x, y) = (pixel.x as i32 + delta.0, pixel.y as i32 + delta.1);
                if !img_size.is_valid(x, y) {
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

fn euclidean_dist(v1: VertexCoord, v2: VertexCoord) -> usize {
    v1.0.abs_diff(v2.0).max(v1.1.abs_diff(v2.1))
}

fn manhattan_dist(v1: VertexCoord, v2: VertexCoord) -> usize {
    v1.0.abs_diff(v2.0) + v1.1.abs_diff(v2.1)
}

#[derive(Clone, Debug)]
pub struct Bounds {
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
}
impl Bounds {
    fn new() -> Self {
        Self {
            x_min: usize::MAX,
            x_max: 0,
            y_min: usize::MAX,
            y_max: 0,
        }
    }
}

pub type RawPath = Vec<VertexCoord>;

#[derive(Clone, Debug)]
pub struct Path {
    pos: RawPath,
    bounds: Bounds,
}

pub type Paths = Vec<Path>;

fn convert_edges_to_paths(edges: &Buffer2D<bool>, frame_index: usize) -> Vec<Path> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "convert_edges_to_paths").entered();

    let mut pathed_vertices = Buffer2D::new(None, edges.size);
    let mut all_paths = Vec::new();

    let min_path_len = frame_min_path_length(frame_index);

    const POSSIBLE_PATHS_PIXELS: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ];

    // Iterate the whole image with 3x3 squares
    // STEP BY => may miss "some" paths for more performances
    for x in (0..edges.size.w - 3).step_by(2) {
        for y in (0..edges.size.h - 3).step_by(2) {
            // Center is an edge, not visited,
            if !edges.get(x + 1, y + 1) || pathed_vertices.get(x + 1, y + 1).is_some() {
                continue;
            }
            // Find valid path-like shapes
            let mut invalid = false;
            let mut side_vertices = Vec::new();

            for &(i, j) in POSSIBLE_PATHS_PIXELS.iter() {
                if edges.get(x + i, y + j) {
                    side_vertices.push((x + i, y + j));
                }
                if pathed_vertices.get(x + i, y + j).is_some() {
                    // TODO Optim May advance cursor by 1 2 or 3
                    invalid = true;
                    break;
                }
            }
            if invalid {
                continue;
            }

            // There are exactly 2 sides vertices, and the dist between the two side vertices is > 1
            let is_valid_path_shape =
                side_vertices.len() == 2 && euclidean_dist(side_vertices[0], side_vertices[1]) > 1;
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
            let mut overriding_path = pathfinding::prelude::astar(
                &path_start,
                |&(x, y)| {
                    let mut successors = Vec::new();
                    for (i, j) in ALL_NEIGHBORS.iter() {
                        let (xi, yj) = (x as i32 + i, y as i32 + j);
                        if edges.is_valid(xi, yj)
                            && pathed_vertices.get(xi as usize, yj as usize).is_none()
                        {
                            let (xi, yj) = (xi as usize, yj as usize);
                            if edges.get(xi, yj) && path_center != (xi, yj) {
                                // Prefer going straight than going diagonally (helps to prevent domains from overlapping)
                                let cost = manhattan_dist((x, y), (xi, yj));
                                successors.push(((xi, yj), cost));
                            }
                        }
                    }
                    successors
                },
                |&p| manhattan_dist(p, path_end),
                |&p| p == path_end,
            );

            match overriding_path {
                // Path does not overrides any other path, simply register it
                Some(mut raw_path) => {
                    // Re-add the center vertex to the path
                    raw_path.0.push(path_center);

                    if raw_path.0.len() < min_path_len {
                        continue;
                    }

                    let path_id = all_paths.len();
                    // Mark pathed vertices as visited
                    let mut bounds = Bounds::new();
                    for v in raw_path.0.iter() {
                        *pathed_vertices.get_mut(v.0, v.1) = Some((path_id, raw_path.1));
                        if v.0 > bounds.x_max {
                            bounds.x_max = v.0;
                        }
                        if v.0 < bounds.x_min {
                            bounds.x_min = v.0;
                        }
                        if v.1 > bounds.y_max {
                            bounds.y_max = v.1;
                        }
                        if v.1 < bounds.y_min {
                            bounds.y_min = v.1;
                        }
                    }

                    all_paths.push((
                        Path {
                            pos: raw_path.0,
                            bounds,
                        },
                        true,
                    ));
                }
                None => {
                    // Try to find a longer path by using already visited vertices
                    overriding_path = pathfinding::prelude::astar(
                        &path_start,
                        |&(x, y)| {
                            let mut successors = Vec::new();
                            for (i, j) in ALL_NEIGHBORS.iter() {
                                let (xi, yj) = (x as i32 + i, y as i32 + j);
                                if edges.is_valid(xi, yj) {
                                    let (xi, yj) = (xi as usize, yj as usize);
                                    if edges.get(xi, yj) && path_center != (xi, yj) {
                                        // Prefer going straight than going diagonally (helps to prevent domains from overlapping)
                                        let cost = manhattan_dist((x, y), (xi, yj));
                                        successors.push(((xi, yj), cost));
                                    }
                                }
                            }
                            successors
                        },
                        |&p| manhattan_dist(p, path_end),
                        |&p| p == path_end,
                    );

                    let Some(mut overriding_path) = overriding_path else {
                        continue;
                    };

                    // Re-add the center vertex to the path
                    overriding_path.0.push(path_center);

                    let length = overriding_path.1;
                    if length < min_path_len {
                        continue;
                    }

                    // Check pathed vertices for longer paths
                    let mut valid = true;
                    let mut invalidated_paths = HashSet::new();
                    for v in overriding_path.0.iter() {
                        if let Some((other_id, other_length)) = pathed_vertices.get(v.0, v.1) {
                            if length <= other_length {
                                valid = false;
                                break;
                            }
                            invalidated_paths.insert(other_id);
                        }
                    }

                    if valid {
                        // Mark pathed vertices as belonging to this path
                        let path_id = all_paths.len();
                        let mut bounds = Bounds::new();
                        for v in overriding_path.0.iter() {
                            *pathed_vertices.get_mut(v.0, v.1) = Some((path_id, length));
                            if v.0 > bounds.x_max {
                                bounds.x_max = v.0;
                            }
                            if v.0 < bounds.x_min {
                                bounds.x_min = v.0;
                            }
                            if v.1 > bounds.y_max {
                                bounds.y_max = v.1;
                            }
                            if v.1 < bounds.y_min {
                                bounds.y_min = v.1;
                            }
                        }
                        all_paths.push((
                            Path {
                                pos: overriding_path.0,
                                bounds,
                            },
                            true,
                        ));

                        // Free vertices used by invalidated paths
                        for invalidated_path in invalidated_paths.iter() {
                            // Mark as invalid
                            all_paths[*invalidated_path].1 = false;
                            for v in all_paths[*invalidated_path].0.pos.iter() {
                                if let Some((id, _)) = pathed_vertices.get(v.0, v.1) {
                                    if id == *invalidated_path {
                                        *pathed_vertices.get_mut(v.0, v.1) = None;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    all_paths
        .iter()
        .filter(|(_path, valid)| *valid)
        .map(|(path, _)| (*path).clone())
        .collect()
}

#[derive(Clone)]
pub struct Domain {
    // Path is in global positions
    path: Path,
    // Local region buffer
    buffer: Buffer2D<bool>,
}
impl Domain {
    fn get_global(&self, x: usize, y: usize) -> bool {
        if x <= self.path.bounds.x_min - 1
            || y <= self.path.bounds.y_min - 1
            || x >= self.path.bounds.x_max + 1
            || y >= self.path.bounds.y_max + 1
        {
            false
        } else {
            self.buffer.get(
                x - (self.path.bounds.x_min - 1),
                y - (self.path.bounds.y_min - 1),
            )
        }
    }
}

fn get_domains_from_paths(mut paths: Paths) -> Vec<Domain> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "get_domains_from_paths").entered();

    let mut domains = Vec::new();
    let mut flood_fill_stack = Vec::new();
    while let Some(path) = paths.pop() {
        // Initialize a full domain. Add a wrapping border of 1 pixel on each side to make sure we have a border not belonging to the domain (=> size + 2). We can always add this border since it exists on the frame.
        let domain_buffer_size = Size {
            w: (path.bounds.x_max - path.bounds.x_min) + 3,
            h: (path.bounds.y_max - path.bounds.y_min) + 3,
        };
        let mut buffer = Buffer2D::new(true, domain_buffer_size);
        // Mark paths vertices as not in the domain
        for p in path.pos.iter() {
            // Convert to local position
            *buffer.get_mut(p.0 - (path.bounds.x_min - 1), p.1 - (path.bounds.y_min - 1)) = false;
        }
        let mut domain = Domain { path, buffer };

        // Flood fill/erase the domain exterior. Working in domain local coordinates
        flood_fill_stack.push(PixelPos { x: 0, y: 0 });
        while let Some(pixel) = flood_fill_stack.pop() {
            *domain.buffer.get_mut(pixel.x as usize, pixel.y as usize) = false;

            for delta in DIRECT_NEIGHBORS.iter() {
                let (x, y) = (pixel.x as i32 + delta.0, pixel.y as i32 + delta.1);
                if !domain.buffer.is_valid(x, y) {
                    continue;
                }
                let pos = PixelPos::new(x as u32, y as u32);
                if domain.buffer.get(pos.x as usize, pos.y as usize) {
                    flood_fill_stack.push(pos);
                }
            }
        }

        domains.push(domain);
    }

    domains
}

fn get_domains_kinds(domains: &Vec<Domain>) -> Vec<DomainKind> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "get_domains_kinds").entered();

    // We know that domains are either fully overlapping or disjoint. We just need to check 1 point

    // Find paths hierarchy (who contains who), then invert orientation for each level
    let mut hierarchies = vec![DomainHierarchy::default(); domains.len()];
    for (i, domain) in domains.iter().enumerate() {
        for (j, other_domain) in domains[i + 1..domains.len()].iter().enumerate() {
            let j = j + i + 1;

            let (p_x, p_y) = (domain.path.pos[0].0, domain.path.pos[0].1);
            let (other_p_x, other_p_y) = (other_domain.path.pos[0].0, other_domain.path.pos[0].1);
            if domain.get_global(other_p_x, other_p_y) && !other_domain.get_global(p_x, p_y) {
                // i C j
                hierarchies[i].children.push(j);
                hierarchies[j].parent = Some(i);
            } else if other_domain.get_global(p_x, p_y) && !domain.get_global(other_p_x, other_p_y)
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

fn get_domains_paths_orientations(domains: &Vec<Domain>) -> Vec<Orientation> {
    #[cfg(feature = "profile_traces")]
    let _span = span!(Level::TRACE, "get_domains_paths_orientations").entered();

    // Paths orientations are != based on the starting shape+location.. See if it lies to the right or left of the path => gives the path orientation.
    let mut orientations = Vec::with_capacity(domains.len());
    for domain in domains.iter() {
        let path = &domain.path.pos;
        let delta_x = (path[1].0 as i32) - (path[0].0 as i32);
        let delta_y = (path[1].1 as i32) - (path[0].1 as i32);
        // Quick & dirty. Might discard left neighbor
        let right_neighbor = if delta_x == 1 && delta_y == -1 {
            (path[0].0 as i32 + 1, path[0].1 as i32)
        } else if delta_x == 1 && delta_y == 1 {
            (path[0].0 as i32, path[0].1 as i32 + 1)
        } else if delta_x == -1 && delta_y == -1 {
            (path[0].0 as i32, path[0].1 as i32 - 1)
        } else if delta_x == -1 && delta_y == 1 {
            (path[0].0 as i32 - 1, path[0].1 as i32)
        } else {
            // delta_x == 0 || delta_y == 0
            (path[0].0 as i32 - delta_y, path[0].1 as i32 + delta_x)
        };

        let path_vertices_orientation =
            if domain.get_global(right_neighbor.0 as usize, right_neighbor.1 as usize) {
                Orientation::CW
            } else {
                Orientation::CCW
            };
        orientations.push(path_vertices_orientation);
    }

    orientations
}
