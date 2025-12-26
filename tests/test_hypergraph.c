/**
 * @file test_hypergraph.c
 * @brief NTTESHGNN - Hypergraph tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ntteshgnn/ntteshgnn.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

#define TEST_PASS(name) printf("PASS: %s\n", name)

/* ============================================================================
 * TESTS
 * ============================================================================ */

int test_graph_creation(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    TEST_ASSERT(g != NULL, "graph creation");
    TEST_ASSERT(g->n_nodes == 0, "initial n_nodes");
    TEST_ASSERT(g->n_edges == 0, "initial n_edges");
    
    nt_graph_set_name(g, "test_graph");
    TEST_ASSERT(strcmp(g->name, "test_graph") == 0, "graph name set");
    
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_creation");
    return 0;
}

int test_graph_nodes(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    TEST_ASSERT(g != NULL, "graph creation");
    
    /* Add tensors as nodes */
    nt_tensor_t* a = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* b = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* c = nt_tensor_new_1d(ctx, NT_F32, 10);
    
    uint32_t idx_a = nt_graph_add_node(g, a);
    uint32_t idx_b = nt_graph_add_node(g, b);
    uint32_t idx_c = nt_graph_add_node(g, c);
    
    TEST_ASSERT(idx_a == 0, "first node index");
    TEST_ASSERT(idx_b == 1, "second node index");
    TEST_ASSERT(idx_c == 2, "third node index");
    TEST_ASSERT(g->n_nodes == 3, "n_nodes after adding");
    
    /* Get node by index */
    nt_tensor_t* retrieved = nt_graph_get_node(g, 1);
    TEST_ASSERT(retrieved == b, "get node by index");
    
    /* Find node */
    int32_t found = nt_graph_find_node(g, c);
    TEST_ASSERT(found == 2, "find node");
    
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_tensor_release(c);
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_nodes");
    return 0;
}

int test_graph_edges(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    TEST_ASSERT(g != NULL, "graph creation");
    
    /* Create tensors */
    nt_tensor_t* a = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* b = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* c = nt_tensor_new_1d(ctx, NT_F32, 10);
    
    nt_graph_add_node(g, a);
    nt_graph_add_node(g, b);
    nt_graph_add_node(g, c);
    
    /* Add compute edge: c = a + b */
    nt_edge_t* e = nt_graph_add_compute(g, 1, c, a, b);  /* op=1 for add */
    TEST_ASSERT(e != NULL, "edge creation");
    TEST_ASSERT(g->n_edges == 1, "n_edges after adding");
    TEST_ASSERT(e->n_inputs == 2, "edge n_inputs");
    TEST_ASSERT(e->n_outputs == 1, "edge n_outputs");
    TEST_ASSERT(e->kind == NT_EDGE_COMPUTE, "edge kind");
    
    /* Add unary edge */
    nt_tensor_t* d = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_graph_add_node(g, d);
    
    nt_edge_t* e2 = nt_graph_add_unary(g, 2, d, c);  /* op=2 for some unary */
    TEST_ASSERT(e2 != NULL, "unary edge creation");
    TEST_ASSERT(e2->n_inputs == 1, "unary edge n_inputs");
    
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_tensor_release(c);
    nt_tensor_release(d);
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_edges");
    return 0;
}

int test_graph_build(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    TEST_ASSERT(g != NULL, "graph creation");
    
    /* Create a simple computation graph */
    nt_tensor_t* a = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* b = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* c = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* d = nt_tensor_new_1d(ctx, NT_F32, 10);
    
    nt_graph_add_node(g, a);
    nt_graph_add_node(g, b);
    nt_graph_add_node(g, c);
    nt_graph_add_node(g, d);
    
    /* c = a + b */
    nt_graph_add_compute(g, 1, c, a, b);
    /* d = relu(c) */
    nt_graph_add_unary(g, 2, d, c);
    
    /* Build execution order */
    nt_graph_build(g);
    TEST_ASSERT(g->exec_order_valid, "exec order valid");
    TEST_ASSERT(g->exec_order != NULL, "exec order allocated");
    TEST_ASSERT(g->exec_order_len == 2, "exec order length");
    
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_tensor_release(c);
    nt_tensor_release(d);
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_build");
    return 0;
}

int test_graph_attention_edge(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    TEST_ASSERT(g != NULL, "graph creation");
    
    /* Create attention tensors */
    int seq_len = 16;
    int d_model = 64;
    int n_heads = 8;
    
    nt_tensor_t* Q = nt_tensor_new_2d(ctx, NT_F32, d_model, seq_len);
    nt_tensor_t* K = nt_tensor_new_2d(ctx, NT_F32, d_model, seq_len);
    nt_tensor_t* V = nt_tensor_new_2d(ctx, NT_F32, d_model, seq_len);
    nt_tensor_t* output = nt_tensor_new_2d(ctx, NT_F32, d_model, seq_len);
    
    /* Add attention hyperedge */
    float scale = 1.0f / sqrtf((float)(d_model / n_heads));
    nt_edge_t* e = nt_graph_add_attention(g, Q, K, V, output, NULL, n_heads, scale);
    
    TEST_ASSERT(e != NULL, "attention edge creation");
    TEST_ASSERT(e->kind == NT_EDGE_ATTENTION, "attention edge kind");
    TEST_ASSERT(e->n_inputs == 3, "attention n_inputs (no mask)");
    TEST_ASSERT(e->n_outputs == 1, "attention n_outputs");
    TEST_ASSERT(e->op_params[0] == n_heads, "attention n_heads param");
    
    nt_tensor_release(Q);
    nt_tensor_release(K);
    nt_tensor_release(V);
    nt_tensor_release(output);
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_attention_edge");
    return 0;
}

int test_graph_print(void) {
    nt_context_t* ctx = nt_context_new();
    TEST_ASSERT(ctx != NULL, "context creation");
    
    nt_graph_t* g = nt_graph_new(ctx);
    nt_graph_set_name(g, "test_print_graph");
    
    /* Add some nodes */
    nt_tensor_t* a = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_t* b = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_set_name(a, "input_a");
    nt_tensor_set_name(b, "input_b");
    
    nt_graph_add_node(g, a);
    nt_graph_add_node(g, b);
    
    /* Add an edge */
    nt_tensor_t* c = nt_tensor_new_1d(ctx, NT_F32, 10);
    nt_tensor_set_name(c, "output_c");
    nt_graph_add_node(g, c);
    nt_graph_add_compute(g, 1, c, a, b);
    
    printf("\n  Graph print output:\n");
    nt_graph_print(g);
    
    nt_tensor_release(a);
    nt_tensor_release(b);
    nt_tensor_release(c);
    nt_graph_free(g);
    nt_context_free(ctx);
    
    TEST_PASS("graph_print");
    return 0;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("NTTESHGNN Hypergraph Tests\n");
    printf("==========================\n\n");
    
    int failed = 0;
    
    failed += test_graph_creation();
    failed += test_graph_nodes();
    failed += test_graph_edges();
    failed += test_graph_build();
    failed += test_graph_attention_edge();
    failed += test_graph_print();
    
    printf("\n");
    if (failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed.\n", failed);
    }
    
    return failed;
}
