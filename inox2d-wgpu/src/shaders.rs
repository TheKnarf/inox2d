#[include_wgsl_oil::include_wgsl_oil("shaders/vertex.wgsl")]
pub mod vertex {}

#[include_wgsl_oil::include_wgsl_oil("shaders/fragment.wgsl")]
pub mod fragment {}

#[include_wgsl_oil::include_wgsl_oil("shaders/mask.wgsl")]
pub mod mask {}
