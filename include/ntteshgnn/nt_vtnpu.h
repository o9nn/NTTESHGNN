/**
 * @file nt_vtnpu.h
 * @brief NTTESHGNN - Virtual Tensor NPU Backend (Layer 7)
 * 
 * Hardware backend for the Virtual Tensor NPU (VTNPU).
 * Maps NTTESHGNN operations to memory-mapped hardware registers
 * and command queues.
 * 
 * The VTNPU is designed for efficient tensor operations with:
 * - Memory-mapped register interface
 * - Command queue for batched operations
 * - On-chip SRAM for fast access
 * - Hardware support for quantized operations
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_VTNPU_H
#define NTTESHGNN_VTNPU_H

#include "nt_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * VTNPU REGISTER MAP (Hardware Interface)
 * ============================================================================ */

/* Base addresses */
#define VTNPU_REG_BASE      0x60000000      /**< Register base address */
#define VTNPU_SRAM_BASE     0x61000000      /**< SRAM base address */
#define VTNPU_CMDQ_BASE     0x62000000      /**< Command queue base address */

/* SRAM size */
#define VTNPU_SRAM_SIZE     (2 * 1024 * 1024)   /**< 2MB SRAM */

/* Control registers (offsets from REG_BASE) */
#define VTNPU_REG_CMD       0x00            /**< Command register */
#define VTNPU_REG_STATUS    0x04            /**< Status register */
#define VTNPU_REG_CONFIG    0x08            /**< Configuration */
#define VTNPU_REG_ERROR     0x0C            /**< Error code */
#define VTNPU_REG_VERSION   0x10            /**< Hardware version */
#define VTNPU_REG_FEATURES  0x14            /**< Feature flags */
#define VTNPU_REG_IRQ_MASK  0x18            /**< Interrupt mask */
#define VTNPU_REG_IRQ_STATUS 0x1C           /**< Interrupt status */

/* Operation registers */
#define VTNPU_REG_OPCODE    0x20            /**< Operation code */
#define VTNPU_REG_FLAGS     0x24            /**< Operation flags */
#define VTNPU_REG_SCALAR0   0x28            /**< Scalar parameter 0 */
#define VTNPU_REG_SCALAR1   0x2C            /**< Scalar parameter 1 */
#define VTNPU_REG_SCALAR2   0x30            /**< Scalar parameter 2 */
#define VTNPU_REG_SCALAR3   0x34            /**< Scalar parameter 3 */

/* Tensor descriptor registers (8 descriptors × 32 bytes each) */
#define VTNPU_REG_TD_BASE   0x100           /**< Tensor descriptor base */
#define VTNPU_REG_TD_STRIDE 0x20            /**< Stride between descriptors */
#define VTNPU_MAX_TD        8               /**< Maximum tensor descriptors */

/* Per-descriptor layout (32 bytes) */
#define VTNPU_TD_ADDR_LO    0x00            /**< Base address low 32 bits */
#define VTNPU_TD_ADDR_HI    0x04            /**< Base address high 32 bits */
#define VTNPU_TD_NE0        0x08            /**< Dimension 0 */
#define VTNPU_TD_NE1        0x0C            /**< Dimension 1 */
#define VTNPU_TD_NE2        0x10            /**< Dimension 2 */
#define VTNPU_TD_NE3        0x14            /**< Dimension 3 */
#define VTNPU_TD_NB0        0x18            /**< Stride 0 (bytes) */
#define VTNPU_TD_DTYPE      0x1C            /**< Data type + layout */

/* Command queue registers */
#define VTNPU_REG_CMDQ_HEAD 0x200           /**< Command queue head */
#define VTNPU_REG_CMDQ_TAIL 0x204           /**< Command queue tail */
#define VTNPU_REG_CMDQ_SIZE 0x208           /**< Command queue size */
#define VTNPU_REG_CMDQ_ADDR 0x20C           /**< Command queue base address */

/* DMA registers */
#define VTNPU_REG_DMA_SRC   0x240           /**< DMA source address */
#define VTNPU_REG_DMA_DST   0x248           /**< DMA destination address */
#define VTNPU_REG_DMA_SIZE  0x250           /**< DMA transfer size */
#define VTNPU_REG_DMA_CTRL  0x254           /**< DMA control */

/* Telemetry registers */
#define VTNPU_REG_MAC_OPS_LO  0x300         /**< MAC operations (low 32 bits) */
#define VTNPU_REG_MAC_OPS_HI  0x304         /**< MAC operations (high 32 bits) */
#define VTNPU_REG_MEM_BW      0x308         /**< Memory bandwidth (bytes/sec) */
#define VTNPU_REG_UTIL        0x30C         /**< Utilization (0-100%) */
#define VTNPU_REG_TEMP        0x310         /**< Temperature (°C × 10) */
#define VTNPU_REG_POWER       0x314         /**< Power consumption (mW) */
#define VTNPU_REG_CYCLES      0x318         /**< Cycle counter */

/* ============================================================================
 * STATUS AND COMMAND VALUES
 * ============================================================================ */

/* Status register bits */
#define VTNPU_STATUS_IDLE       (1 << 0)    /**< NPU is idle */
#define VTNPU_STATUS_BUSY       (1 << 1)    /**< NPU is busy */
#define VTNPU_STATUS_ERROR      (1 << 2)    /**< Error occurred */
#define VTNPU_STATUS_CMDQ_EMPTY (1 << 3)    /**< Command queue empty */
#define VTNPU_STATUS_CMDQ_FULL  (1 << 4)    /**< Command queue full */
#define VTNPU_STATUS_DMA_BUSY   (1 << 5)    /**< DMA in progress */

/* Command register values */
#define VTNPU_CMD_NOP           0x00        /**< No operation */
#define VTNPU_CMD_EXEC          0x01        /**< Execute operation */
#define VTNPU_CMD_EXEC_CMDQ     0x02        /**< Execute command queue */
#define VTNPU_CMD_RESET         0x03        /**< Reset NPU */
#define VTNPU_CMD_ABORT         0x04        /**< Abort current operation */
#define VTNPU_CMD_DMA_START     0x10        /**< Start DMA transfer */
#define VTNPU_CMD_DMA_ABORT     0x11        /**< Abort DMA transfer */

/* Error codes */
#define VTNPU_ERR_NONE          0x00        /**< No error */
#define VTNPU_ERR_INVALID_OP    0x01        /**< Invalid operation */
#define VTNPU_ERR_INVALID_DTYPE 0x02        /**< Unsupported data type */
#define VTNPU_ERR_INVALID_SHAPE 0x03        /**< Invalid tensor shape */
#define VTNPU_ERR_OUT_OF_SRAM   0x04        /**< Out of SRAM */
#define VTNPU_ERR_DMA_FAIL      0x05        /**< DMA transfer failed */
#define VTNPU_ERR_TIMEOUT       0x06        /**< Operation timeout */

/* Feature flags */
#define VTNPU_FEAT_FP16         (1 << 0)    /**< FP16 support */
#define VTNPU_FEAT_BF16         (1 << 1)    /**< BF16 support */
#define VTNPU_FEAT_INT8         (1 << 2)    /**< INT8 support */
#define VTNPU_FEAT_Q4           (1 << 3)    /**< Q4 quantization */
#define VTNPU_FEAT_Q8           (1 << 4)    /**< Q8 quantization */
#define VTNPU_FEAT_FLASH_ATTN   (1 << 5)    /**< Flash attention */
#define VTNPU_FEAT_SPARSE       (1 << 6)    /**< Sparse operations */

/* ============================================================================
 * COMMAND STRUCTURE
 * ============================================================================ */

/**
 * @brief VTNPU command structure (24 bytes, packed)
 */
typedef struct NT_PACKED vtnpu_cmd {
    uint16_t    opcode;             /**< NT_OP_* code */
    uint16_t    flags;              /**< Operation flags */
    
    uint8_t     src0_td;            /**< Source 0 tensor descriptor index */
    uint8_t     src1_td;            /**< Source 1 tensor descriptor index */
    uint8_t     dst_td;             /**< Destination tensor descriptor index */
    uint8_t     aux_td;             /**< Auxiliary tensor descriptor index */
    
    int32_t     params[4];          /**< Operation-specific parameters */
    
} vtnpu_cmd_t;

_Static_assert(sizeof(vtnpu_cmd_t) == 24, "vtnpu_cmd_t must be 24 bytes");

/* Command queue size */
#define VTNPU_CMDQ_MAX_SIZE     1024        /**< Maximum commands in queue */

/* ============================================================================
 * TENSOR DESCRIPTOR
 * ============================================================================ */

/**
 * @brief VTNPU tensor descriptor (matches hardware layout)
 */
typedef struct NT_PACKED vtnpu_tensor_desc {
    uint64_t    addr;               /**< Base address */
    int32_t     ne[4];              /**< Shape (up to 4D) */
    int32_t     nb[4];              /**< Strides in bytes */
    uint16_t    dtype;              /**< Data type */
    uint16_t    layout;             /**< Memory layout */
} vtnpu_tensor_desc_t;

/* ============================================================================
 * BACKEND CONTEXT
 * ============================================================================ */

/**
 * @brief VTNPU backend context
 */
typedef struct vtnpu_backend_ctx {
    /* Hardware interface */
    volatile uint32_t*  regs;           /**< Register base (memory-mapped) */
    void*               sram;           /**< SRAM mapping */
    size_t              sram_size;      /**< SRAM size */
    
    /* SRAM allocator */
    size_t              sram_used;      /**< SRAM bytes used */
    size_t              sram_peak;      /**< Peak SRAM usage */
    
    /* Command queue */
    vtnpu_cmd_t*        cmdq;           /**< Command queue buffer */
    uint32_t            cmdq_head;      /**< Queue head index */
    uint32_t            cmdq_tail;      /**< Queue tail index */
    uint32_t            cmdq_size;      /**< Queue capacity */
    
    /* Tensor descriptor cache */
    nt_tensor_t*        td_tensors[VTNPU_MAX_TD];   /**< Tensor in each TD slot */
    uint8_t             td_used;        /**< Bitmask of used TD slots */
    
    /* Telemetry */
    uint64_t            total_mac_ops;  /**< Total MAC operations */
    uint64_t            total_bytes;    /**< Total bytes transferred */
    uint64_t            total_cycles;   /**< Total cycles */
    
    /* Configuration */
    bool                is_simulated;   /**< Running in simulation mode */
    int                 timeout_ms;     /**< Operation timeout */
    
    /* Debug */
    bool                debug_mode;     /**< Enable debug output */
    FILE*               debug_file;     /**< Debug output file */
    
} vtnpu_backend_ctx_t;

/* ============================================================================
 * BACKEND LIFECYCLE
 * ============================================================================ */

/**
 * @brief Create VTNPU backend
 * @return Backend instance
 */
nt_backend_t* nt_backend_vtnpu_create(void);

/**
 * @brief Create simulated VTNPU backend (for testing)
 * @return Simulated backend instance
 */
nt_backend_t* nt_backend_vtnpu_simulated(void);

/**
 * @brief Initialize VTNPU hardware
 * @param ctx Backend context
 * @return 0 on success
 */
int vtnpu_init(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Shutdown VTNPU hardware
 * @param ctx Backend context
 */
void vtnpu_shutdown(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Reset VTNPU to initial state
 * @param ctx Backend context
 */
void vtnpu_reset(vtnpu_backend_ctx_t* ctx);

/* ============================================================================
 * LOW-LEVEL REGISTER ACCESS
 * ============================================================================ */

/**
 * @brief Write 32-bit register
 */
NT_INLINE void vtnpu_write_reg32(vtnpu_backend_ctx_t* ctx, uint32_t offset, uint32_t value) {
    ctx->regs[offset / 4] = value;
}

/**
 * @brief Read 32-bit register
 */
NT_INLINE uint32_t vtnpu_read_reg32(vtnpu_backend_ctx_t* ctx, uint32_t offset) {
    return ctx->regs[offset / 4];
}

/**
 * @brief Write 64-bit register (as two 32-bit writes)
 */
void vtnpu_write_reg64(vtnpu_backend_ctx_t* ctx, uint32_t offset, uint64_t value);

/**
 * @brief Read 64-bit register
 */
uint64_t vtnpu_read_reg64(vtnpu_backend_ctx_t* ctx, uint32_t offset);

/* ============================================================================
 * TENSOR DESCRIPTOR MANAGEMENT
 * ============================================================================ */

/**
 * @brief Allocate tensor descriptor slot
 * @param ctx Backend context
 * @param t Tensor to bind
 * @return TD index or -1 if full
 */
int vtnpu_alloc_td(vtnpu_backend_ctx_t* ctx, nt_tensor_t* t);

/**
 * @brief Free tensor descriptor slot
 * @param ctx Backend context
 * @param td_idx TD index to free
 */
void vtnpu_free_td(vtnpu_backend_ctx_t* ctx, int td_idx);

/**
 * @brief Write tensor descriptor to hardware
 * @param ctx Backend context
 * @param td_idx TD index
 * @param t Tensor
 */
void vtnpu_write_td(vtnpu_backend_ctx_t* ctx, int td_idx, const nt_tensor_t* t);

/**
 * @brief Find TD slot for tensor
 * @return TD index or -1 if not found
 */
int vtnpu_find_td(vtnpu_backend_ctx_t* ctx, const nt_tensor_t* t);

/* ============================================================================
 * COMMAND QUEUE
 * ============================================================================ */

/**
 * @brief Submit command to queue
 * @param ctx Backend context
 * @param cmd Command to submit
 * @return 0 on success
 */
int vtnpu_submit_cmd(vtnpu_backend_ctx_t* ctx, const vtnpu_cmd_t* cmd);

/**
 * @brief Flush command queue (execute all pending)
 * @param ctx Backend context
 */
void vtnpu_flush_cmdq(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Wait for NPU to become idle
 * @param ctx Backend context
 * @return 0 on success, -1 on timeout
 */
int vtnpu_wait_idle(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Check if command queue is empty
 */
bool vtnpu_cmdq_empty(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Check if command queue is full
 */
bool vtnpu_cmdq_full(vtnpu_backend_ctx_t* ctx);

/* ============================================================================
 * DATA TRANSFER
 * ============================================================================ */

/**
 * @brief Upload data to VTNPU SRAM
 * @param ctx Backend context
 * @param offset Offset in SRAM
 * @param data Source data
 * @param size Size in bytes
 * @return 0 on success
 */
int vtnpu_upload(vtnpu_backend_ctx_t* ctx, size_t offset, const void* data, size_t size);

/**
 * @brief Download data from VTNPU SRAM
 * @param ctx Backend context
 * @param offset Offset in SRAM
 * @param data Destination buffer
 * @param size Size in bytes
 * @return 0 on success
 */
int vtnpu_download(vtnpu_backend_ctx_t* ctx, size_t offset, void* data, size_t size);

/**
 * @brief Allocate SRAM buffer
 * @param ctx Backend context
 * @param size Size in bytes
 * @param align Alignment
 * @return Offset in SRAM or -1 on failure
 */
int64_t vtnpu_sram_alloc(vtnpu_backend_ctx_t* ctx, size_t size, size_t align);

/**
 * @brief Free SRAM buffer
 */
void vtnpu_sram_free(vtnpu_backend_ctx_t* ctx, int64_t offset, size_t size);

/**
 * @brief Reset SRAM allocator
 */
void vtnpu_sram_reset(vtnpu_backend_ctx_t* ctx);

/* ============================================================================
 * OPERATION MAPPING
 * ============================================================================ */

/**
 * @brief Map NT_OP_* to VTNPU hardware opcode
 */
NT_INLINE uint16_t vtnpu_map_opcode(nt_op_t op) {
    /* Direct mapping - VTNPU uses same opcode space */
    return (uint16_t)op;
}

/**
 * @brief Check if operation is supported in hardware
 * @param op Operation code
 * @param dtype Data type
 * @return true if supported
 */
bool vtnpu_op_supported(nt_op_t op, nt_dtype_t dtype);

/**
 * @brief Get estimated cycles for operation
 */
uint64_t vtnpu_estimate_cycles(nt_op_t op, const nt_tensor_t** inputs, int n_inputs);

/* ============================================================================
 * HIGH-LEVEL OPERATIONS
 * ============================================================================ */

/**
 * @brief Execute operation on VTNPU
 * @param ctx Backend context
 * @param op Operation code
 * @param inputs Input tensors
 * @param n_inputs Number of inputs
 * @param output Output tensor
 * @param params Operation parameters
 */
void vtnpu_compute(vtnpu_backend_ctx_t* ctx, nt_op_t op,
                   const nt_tensor_t** inputs, int n_inputs,
                   nt_tensor_t* output, const nt_op_params_t* params);

/**
 * @brief Execute graph on VTNPU
 */
void vtnpu_graph_compute(vtnpu_backend_ctx_t* ctx, nt_graph_t* g);

/* ============================================================================
 * TELEMETRY
 * ============================================================================ */

/**
 * @brief VTNPU telemetry data
 */
typedef struct vtnpu_telemetry {
    uint64_t    mac_ops;            /**< MAC operations */
    uint64_t    memory_bytes;       /**< Memory transferred */
    uint64_t    cycles;             /**< Cycles elapsed */
    uint32_t    utilization;        /**< Utilization percentage */
    uint32_t    temperature;        /**< Temperature (°C × 10) */
    uint32_t    power_mw;           /**< Power consumption (mW) */
    float       gflops;             /**< Effective GFLOPS */
    float       bandwidth_gbps;     /**< Memory bandwidth (GB/s) */
} vtnpu_telemetry_t;

/**
 * @brief Read telemetry from hardware
 */
void vtnpu_read_telemetry(vtnpu_backend_ctx_t* ctx, vtnpu_telemetry_t* telem);

/**
 * @brief Reset telemetry counters
 */
void vtnpu_reset_telemetry(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Print telemetry summary
 */
void vtnpu_print_telemetry(const vtnpu_telemetry_t* telem);

/* ============================================================================
 * DEBUG AND DIAGNOSTICS
 * ============================================================================ */

/**
 * @brief Enable debug mode
 */
void vtnpu_set_debug(vtnpu_backend_ctx_t* ctx, bool enable, FILE* output);

/**
 * @brief Dump register state
 */
void vtnpu_dump_regs(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Dump tensor descriptors
 */
void vtnpu_dump_tds(vtnpu_backend_ctx_t* ctx);

/**
 * @brief Run self-test
 * @return 0 on success
 */
int vtnpu_self_test(vtnpu_backend_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_VTNPU_H */
