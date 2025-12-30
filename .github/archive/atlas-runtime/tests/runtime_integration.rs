use atlas_core::AtlasRatio;
use atlas_isa::{BlockDim, ClassMask, GridDim, KernelMetadata, LaunchConfig, ResonanceClass};
use atlas_runtime::{enumerate_devices, ContextDesc, Kernel, PhaseAdvance, QueueDesc};

fn launch_config() -> LaunchConfig {
    LaunchConfig::new(GridDim::new(1, 1, 1), BlockDim::new(1, 1, 1), 0)
}

#[test]
fn runtime_executes_kernel_and_updates_resonance() {
    let device = enumerate_devices().pop().expect("device");
    let ctx = device.create_context(ContextDesc::default());
    let queue = ctx.create_queue(QueueDesc::default());

    let mut metadata = KernelMetadata::new("resonance_add");
    metadata.classes_mask = ClassMask::single(ResonanceClass::new(0).unwrap());

    let kernel = Kernel::new(metadata, |invocation| {
        invocation.space().resonance().add(0, AtlasRatio::new_raw(1, 1))?;
        Ok(())
    })
    .expect("kernel registered");

    let mut burst = ctx.begin_burst();
    burst.enqueue(kernel, launch_config());

    queue.submit(burst).expect("burst submission");

    ctx.with_space(|space| {
        assert_eq!(space.resonance().get(0), AtlasRatio::new_raw(1, 1));
    });
}

#[test]
fn runtime_rejects_phase_window_violation() {
    let device = enumerate_devices().pop().expect("device");
    let ctx = device.create_context(ContextDesc::default());
    let queue = ctx.create_queue(QueueDesc::default());

    let mut metadata = KernelMetadata::new("phase_violation");
    metadata.classes_mask = ClassMask::single(ResonanceClass::new(0).unwrap());
    metadata.phase.begin = 10;
    metadata.phase.span = 1;

    let kernel = Kernel::new(metadata, |_| Ok(())).expect("kernel registered");

    let mut burst = ctx.begin_burst();
    burst.enqueue(kernel, launch_config());

    let err = queue.submit(burst).expect_err("phase window violation expected");

    match err {
        atlas_runtime::AtlasError::PhaseWindow { .. } => {}
        other => panic!("unexpected error: {other:?}", other = other),
    }
}

#[test]
fn runtime_advances_phase_after_burst() {
    let device = enumerate_devices().pop().expect("device");
    let ctx = device.create_context(ContextDesc::default());
    let queue = ctx.create_queue(QueueDesc::default());

    let mut metadata = KernelMetadata::new("phase_advance");
    metadata.classes_mask = ClassMask::single(ResonanceClass::new(0).unwrap());

    let kernel = Kernel::new(metadata, |_| Ok(())).expect("kernel registered");

    let mut burst = ctx.begin_burst();
    burst.enqueue(kernel, launch_config());
    burst.set_phase_advance(PhaseAdvance::new(3));

    queue.submit(burst).expect("burst submission");

    ctx.with_space(|space| {
        assert_eq!(space.phase().get(), 3);
    });
}
