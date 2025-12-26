/**
 * @file nt_typesystem.h
 * @brief NTTESHGNN - Algebraic Type System (Layer 2)
 * 
 * A rich type system for tensors that goes beyond simple dtype checking.
 * Supports:
 * - Shape constraints (fixed, symbolic, broadcast)
 * - Sum types (tensor | nested_tensor | sparse_tensor)
 * - Product types (tuples of tensors)
 * - Dependent types (shape depends on runtime values)
 * - Type inference and checking at graph construction time
 * 
 * Design principles:
 * - Catch errors at graph construction, not execution
 * - Support automatic broadcasting and shape inference
 * - Enable optimization through type information
 * 
 * @author NTTESHGNN Team
 * @version 0.1.0
 */

#ifndef NTTESHGNN_TYPESYSTEM_H
#define NTTESHGNN_TYPESYSTEM_H

#include "nt_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TYPE KINDS
 * ============================================================================ */

/**
 * @brief Kind of type in the type system
 */
typedef enum nt_type_kind {
    NT_TYPE_TENSOR,         /**< Regular tensor type */
    NT_TYPE_NESTED,         /**< Nested tensor type */
    NT_TYPE_SPARSE,         /**< Sparse tensor type */
    NT_TYPE_TUPLE,          /**< Tuple of types (product type) */
    NT_TYPE_UNION,          /**< Union of types (sum type) */
    NT_TYPE_OPTIONAL,       /**< Optional type (T | None) */
    NT_TYPE_ANY,            /**< Any type (dynamic) */
    NT_TYPE_VOID,           /**< Void type (no value) */
} nt_type_kind_t;

/* ============================================================================
 * DIMENSION CONSTRAINTS
 * ============================================================================ */

/**
 * @brief Kind of dimension constraint
 */
typedef enum nt_dim_kind {
    NT_DIM_FIXED,           /**< Fixed size (e.g., 512) */
    NT_DIM_SYMBOLIC,        /**< Symbolic size (e.g., N, M) */
    NT_DIM_BROADCAST,       /**< Broadcastable (1 or matching) */
    NT_DIM_DYNAMIC,         /**< Any size (runtime determined) */
    NT_DIM_RANGE,           /**< Size in range [min, max] */
    NT_DIM_MULTIPLE,        /**< Multiple of a value */
} nt_dim_kind_t;

/**
 * @brief Dimension constraint
 */
typedef struct nt_dim_constraint {
    nt_dim_kind_t   kind;
    union {
        int32_t     fixed;          /**< For NT_DIM_FIXED */
        uint32_t    symbol_id;      /**< For NT_DIM_SYMBOLIC */
        struct {
            int32_t min;
            int32_t max;
        } range;                    /**< For NT_DIM_RANGE */
        int32_t     multiple_of;    /**< For NT_DIM_MULTIPLE */
    };
} nt_dim_constraint_t;

/* Convenience constructors */
#define NT_DIM_FIXED_VAL(n)     ((nt_dim_constraint_t){.kind = NT_DIM_FIXED, .fixed = (n)})
#define NT_DIM_SYM(id)          ((nt_dim_constraint_t){.kind = NT_DIM_SYMBOLIC, .symbol_id = (id)})
#define NT_DIM_ANY              ((nt_dim_constraint_t){.kind = NT_DIM_DYNAMIC})
#define NT_DIM_BROADCAST_VAL    ((nt_dim_constraint_t){.kind = NT_DIM_BROADCAST})

/* ============================================================================
 * SHAPE CONSTRAINTS
 * ============================================================================ */

/**
 * @brief Shape constraint for tensor types
 */
typedef struct nt_shape_constraint {
    uint8_t             ndim;                       /**< Number of dimensions */
    nt_dim_constraint_t dims[NT_MAX_DIMS];          /**< Constraint per dimension */
    bool                allow_broadcast;            /**< Allow broadcasting */
} nt_shape_constraint_t;

/* ============================================================================
 * TYPE STRUCTURE
 * ============================================================================ */

/**
 * @brief Type in the type system
 */
struct nt_type {
    nt_type_kind_t          kind;           /**< Type kind */
    
    /* For tensor types */
    nt_dtype_t              dtype;          /**< Data type */
    nt_shape_constraint_t   shape;          /**< Shape constraint */
    nt_device_t             device;         /**< Device constraint (or -1 for any) */
    
    /* For compound types */
    struct nt_type**        subtypes;       /**< Sub-types (for tuple/union) */
    uint8_t                 n_subtypes;     /**< Number of sub-types */
    
    /* Type metadata */
    const char*             name;           /**< Optional type name */
    uint32_t                type_id;        /**< Unique type ID */
    
    /* Reference counting */
    _Atomic int32_t         refcount;
};

/* ============================================================================
 * TYPE CREATION
 * ============================================================================ */

/**
 * @brief Create a tensor type with shape constraint
 * @param dtype Data type
 * @param ndim Number of dimensions
 * @param dims Dimension constraints
 * @return New type
 */
nt_type_t* nt_type_tensor(nt_dtype_t dtype, uint8_t ndim, const nt_dim_constraint_t* dims);

/**
 * @brief Create a tensor type with fixed shape
 */
nt_type_t* nt_type_tensor_fixed(nt_dtype_t dtype, uint8_t ndim, const int32_t* shape);

/**
 * @brief Create a tensor type with dynamic shape
 */
nt_type_t* nt_type_tensor_dynamic(nt_dtype_t dtype, uint8_t ndim);

/**
 * @brief Create a scalar type
 */
nt_type_t* nt_type_scalar(nt_dtype_t dtype);

/**
 * @brief Create a vector type
 */
nt_type_t* nt_type_vector(nt_dtype_t dtype, int32_t size);

/**
 * @brief Create a matrix type
 */
nt_type_t* nt_type_matrix(nt_dtype_t dtype, int32_t rows, int32_t cols);

/**
 * @brief Create a nested tensor type
 */
nt_type_t* nt_type_nested(nt_type_t* inner_type);

/**
 * @brief Create a sparse tensor type
 */
nt_type_t* nt_type_sparse(nt_dtype_t dtype, uint8_t ndim, nt_layout_t sparse_format);

/**
 * @brief Create a tuple type (product type)
 */
nt_type_t* nt_type_tuple(nt_type_t** types, uint8_t n_types);

/**
 * @brief Create a union type (sum type)
 */
nt_type_t* nt_type_union(nt_type_t** types, uint8_t n_types);

/**
 * @brief Create an optional type
 */
nt_type_t* nt_type_optional(nt_type_t* inner_type);

/**
 * @brief Create the any type
 */
nt_type_t* nt_type_any(void);

/* ============================================================================
 * TYPE LIFECYCLE
 * ============================================================================ */

/**
 * @brief Retain type
 */
nt_type_t* nt_type_retain(nt_type_t* type);

/**
 * @brief Release type
 */
void nt_type_release(nt_type_t* type);

/**
 * @brief Clone type
 */
nt_type_t* nt_type_clone(const nt_type_t* type);

/* ============================================================================
 * TYPE CHECKING
 * ============================================================================ */

/**
 * @brief Type checking error codes
 */
typedef enum nt_type_error {
    NT_TYPE_OK = 0,                     /**< Type check passed */
    NT_TYPE_ERR_KIND_MISMATCH,          /**< Type kinds don't match */
    NT_TYPE_ERR_DTYPE_MISMATCH,         /**< Data types don't match */
    NT_TYPE_ERR_NDIM_MISMATCH,          /**< Number of dimensions don't match */
    NT_TYPE_ERR_SHAPE_MISMATCH,         /**< Shape constraint violated */
    NT_TYPE_ERR_DEVICE_MISMATCH,        /**< Device constraint violated */
    NT_TYPE_ERR_CONSTRAINT_VIOLATION,   /**< Generic constraint violation */
    NT_TYPE_ERR_BROADCAST_FAIL,         /**< Broadcasting not possible */
    NT_TYPE_ERR_NESTED_DEPTH,           /**< Nested depth exceeded */
    NT_TYPE_ERR_SYMBOL_CONFLICT,        /**< Symbolic dimension conflict */
} nt_type_error_t;

/**
 * @brief Check if tensor matches type
 * @param t Tensor to check
 * @param expected Expected type
 * @return NT_TYPE_OK if matches, error code otherwise
 */
nt_type_error_t nt_type_check(const nt_tensor_t* t, const nt_type_t* expected);

/**
 * @brief Check if type A is subtype of type B
 */
bool nt_type_is_subtype(const nt_type_t* a, const nt_type_t* b);

/**
 * @brief Check if two types are equal
 */
bool nt_type_equal(const nt_type_t* a, const nt_type_t* b);

/**
 * @brief Check if two types are compatible (can be unified)
 */
bool nt_type_compatible(const nt_type_t* a, const nt_type_t* b);

/**
 * @brief Get human-readable error message
 */
const char* nt_type_error_str(nt_type_error_t err);

/* ============================================================================
 * TYPE INFERENCE
 * ============================================================================ */

/**
 * @brief Infer type from tensor
 */
nt_type_t* nt_type_infer(const nt_tensor_t* t);

/**
 * @brief Infer result type of binary operation
 */
nt_type_t* nt_type_infer_binary(const nt_type_t* a, const nt_type_t* b, uint16_t op);

/**
 * @brief Infer result type of unary operation
 */
nt_type_t* nt_type_infer_unary(const nt_type_t* input, uint16_t op);

/**
 * @brief Infer broadcast result shape
 */
nt_status_t nt_type_broadcast_shape(const nt_type_t* a, const nt_type_t* b,
                                     int32_t* out_shape, uint8_t* out_ndim);

/* ============================================================================
 * OPERATION SIGNATURES
 * ============================================================================ */

/**
 * @brief Operation type signature
 */
typedef struct nt_op_signature {
    const char*     name;               /**< Operation name */
    uint16_t        op_code;            /**< Operation code */
    
    /* Input types */
    nt_type_t**     input_types;        /**< Expected input types */
    uint8_t         n_inputs;           /**< Number of inputs */
    uint8_t         n_required;         /**< Number of required inputs */
    
    /* Output type */
    nt_type_t*      output_type;        /**< Output type (or NULL for inferred) */
    
    /* Type inference function (if output depends on inputs) */
    nt_type_t*      (*infer_output)(const nt_type_t** inputs, uint8_t n_inputs);
    
    /* Constraints */
    bool            same_dtype;         /**< Inputs must have same dtype */
    bool            same_device;        /**< Inputs must be on same device */
    bool            broadcastable;      /**< Inputs can be broadcast */
    
} nt_op_signature_t;

/**
 * @brief Check if operation can be applied to inputs
 */
nt_type_error_t nt_op_type_check(const nt_op_signature_t* op,
                                  const nt_tensor_t** inputs,
                                  uint8_t n_inputs);

/**
 * @brief Get operation signature by op code
 */
const nt_op_signature_t* nt_op_get_signature(uint16_t op_code);

/**
 * @brief Register custom operation signature
 */
nt_status_t nt_op_register_signature(const nt_op_signature_t* sig);

/* ============================================================================
 * SYMBOL TABLE (for symbolic dimensions)
 * ============================================================================ */

/**
 * @brief Symbol table for tracking symbolic dimension bindings
 */
typedef struct nt_symbol_table {
    uint32_t*       symbol_ids;         /**< Symbol IDs */
    int32_t*        values;             /**< Bound values (-1 for unbound) */
    uint32_t        n_symbols;          /**< Number of symbols */
    uint32_t        capacity;           /**< Table capacity */
} nt_symbol_table_t;

/**
 * @brief Create symbol table
 */
nt_symbol_table_t* nt_symbol_table_new(void);

/**
 * @brief Free symbol table
 */
void nt_symbol_table_free(nt_symbol_table_t* table);

/**
 * @brief Bind symbol to value
 */
nt_status_t nt_symbol_bind(nt_symbol_table_t* table, uint32_t symbol_id, int32_t value);

/**
 * @brief Get symbol value
 */
int32_t nt_symbol_get(const nt_symbol_table_t* table, uint32_t symbol_id);

/**
 * @brief Check if symbol is bound
 */
bool nt_symbol_is_bound(const nt_symbol_table_t* table, uint32_t symbol_id);

/**
 * @brief Create new symbol ID
 */
uint32_t nt_symbol_new(nt_symbol_table_t* table);

/* ============================================================================
 * TYPE UTILITIES
 * ============================================================================ */

/**
 * @brief Print type to string
 */
void nt_type_print(const nt_type_t* type);

/**
 * @brief Get type string representation
 */
const char* nt_type_str(const nt_type_t* type);

/**
 * @brief Common type patterns
 */
extern nt_type_t* NT_TYPE_F32_SCALAR;
extern nt_type_t* NT_TYPE_F32_VECTOR;
extern nt_type_t* NT_TYPE_F32_MATRIX;
extern nt_type_t* NT_TYPE_F32_ANY;
extern nt_type_t* NT_TYPE_I64_SCALAR;
extern nt_type_t* NT_TYPE_BOOL_SCALAR;

/**
 * @brief Initialize common types
 */
void nt_type_init_common(void);

/**
 * @brief Cleanup common types
 */
void nt_type_cleanup_common(void);

#ifdef __cplusplus
}
#endif

#endif /* NTTESHGNN_TYPESYSTEM_H */
