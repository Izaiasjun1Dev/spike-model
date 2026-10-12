//! MTLIOCommandQueue backend: Metal loads expert bytes from the file straight
//! into MTLBuffers (shared storage), with no CPU copy in our code.
//! Requires macOS 14+ (newIOFileHandleWithURL:error:).

use std::path::Path;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLIOCommandBuffer, MTLIOCommandQueue,
    MTLIOCommandQueueDescriptor, MTLIOCommandQueueType, MTLIOFileHandle, MTLIOPriority,
    MTLIOStatus, MTLResourceOptions,
};

use crate::backend::Backend;

pub struct MetalIo {
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
    handle: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
    buffers: Vec<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    in_flight: Vec<Option<Retained<ProtocolObject<dyn MTLIOCommandBuffer>>>>,
    expert_bytes: usize,
}

impl MetalIo {
    pub fn new(
        path: &Path,
        slots: usize,
        topk: usize,
        expert_bytes: usize,
    ) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        eprintln!("metal device: {}", device.name());

        let desc = MTLIOCommandQueueDescriptor::new();
        desc.setType(MTLIOCommandQueueType::Concurrent);
        desc.setPriority(MTLIOPriority::High);
        let queue = device
            .newIOCommandQueueWithDescriptor_error(&desc)
            .map_err(|e| format!("IO queue: {}", e.localizedDescription()))?;

        let abs = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(&abs.to_string_lossy()));
        let handle = device
            .newIOFileHandleWithURL_error(&url)
            .map_err(|e| format!("IO file handle: {}", e.localizedDescription()))?;

        let mut buffers = Vec::with_capacity(slots);
        for _ in 0..slots {
            let mut row = Vec::with_capacity(topk);
            for _ in 0..topk {
                row.push(
                    device
                        .newBufferWithLength_options(
                            expert_bytes,
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or("MTLBuffer allocation failed")?,
                );
            }
            buffers.push(row);
        }

        Ok(Self {
            queue,
            handle,
            buffers,
            in_flight: vec![None; slots],
            expert_bytes,
        })
    }
}

impl Backend for MetalIo {
    fn submit(&mut self, slot: usize, offsets: &[u64]) -> Result<(), String> {
        let cb = self.queue.commandBuffer();
        for (i, &offset) in offsets.iter().enumerate() {
            // Safety: buffers live as long as `self`, and the size fits each buffer.
            unsafe {
                cb.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    &self.buffers[slot][i],
                    0,
                    self.expert_bytes,
                    &self.handle,
                    offset as usize,
                )
            };
        }
        cb.commit();
        self.in_flight[slot] = Some(cb);
        Ok(())
    }

    fn wait(&mut self, slot: usize) -> Result<(), String> {
        let cb = self.in_flight[slot].take().ok_or("wait on idle slot")?;
        cb.waitUntilCompleted();
        if cb.status() == MTLIOStatus::Complete {
            Ok(())
        } else {
            let msg = cb.error().map(|e| e.localizedDescription().to_string());
            Err(format!(
                "IO command buffer status {:?}: {}",
                cb.status().0,
                msg.unwrap_or_default()
            ))
        }
    }

    fn buffer(&self, slot: usize, idx: usize) -> &[u8] {
        let buf = &self.buffers[slot][idx];
        unsafe {
            std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, self.expert_bytes)
        }
    }
}
