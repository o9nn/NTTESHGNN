/**
 * @file nt_hypergraph.c
 * @brief NTTESHGNN - Hypergraph implementation
 */

#include "ntteshgnn/nt_hypergraph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * GRAPH LIFECYCLE
 * ============================================================================ */

nt_graph_t* nt_graph_new(nt_context_t* ctx) {
    nt_graph_t* g = (nt_graph_t*)calloc(1, sizeof(nt_graph_t));
    if (!g) return NULL;
    
    g->ctx = ctx;
    g->graph_id = 0;
    g->n_nodes = 0;
    g->nodes_capacity = 64;
    g->n_edges = 0;
    
    g->nodes = (nt_tensor_t**)calloc(g->nodes_capacity, sizeof(nt_tensor_t*));
    if (!g->nodes) {
        free(g);
        return NULL;
    }
    
    g->edges_head = NULL;
    g->edges_tail = NULL;
    g->exec_order = NULL;
    g->exec_order_len = 0;
    g->exec_order_valid = false;
    
    g->is_training = false;
    g->enable_grad = false;
    
    return g;
}

void nt_graph_free(nt_graph_t* g) {
    if (!g) return;
    
    /* Free edges */
    nt_edge_t* e = g->edges_head;
    while (e) {
        nt_edge_t* next = e->next;
        
        if (e->inputs) free(e->inputs);
        if (e->outputs) free(e->outputs);
        if (e->edge_weight) nt_tensor_release(e->edge_weight);
        if (e->edge_attr) nt_tensor_release(e->edge_attr);
        free(e);
        
        e = next;
    }
    
    if (g->nodes) free(g->nodes);
    if (g->exec_order) free(g->exec_order);
    if (g->checkpoints) free(g->checkpoints);
    
    free(g);
}

void nt_graph_clear(nt_graph_t* g) {
    if (!g) return;
    
    /* Free edges */
    nt_edge_t* e = g->edges_head;
    while (e) {
        nt_edge_t* next = e->next;
        
        if (e->inputs) free(e->inputs);
        if (e->outputs) free(e->outputs);
        if (e->edge_weight) nt_tensor_release(e->edge_weight);
        if (e->edge_attr) nt_tensor_release(e->edge_attr);
        free(e);
        
        e = next;
    }
    
    g->edges_head = NULL;
    g->edges_tail = NULL;
    g->n_edges = 0;
    g->n_nodes = 0;
    g->exec_order_valid = false;
}

void nt_graph_set_name(nt_graph_t* g, const char* name) {
    if (!g || !name) return;
    strncpy(g->name, name, NT_MAX_NAME - 1);
    g->name[NT_MAX_NAME - 1] = '\0';
}

/* ============================================================================
 * NODE MANAGEMENT
 * ============================================================================ */

uint32_t nt_graph_add_node(nt_graph_t* g, nt_tensor_t* t) {
    if (!g || !t) return (uint32_t)-1;
    
    /* Expand capacity if needed */
    if (g->n_nodes >= g->nodes_capacity) {
        uint32_t new_cap = g->nodes_capacity * 2;
        nt_tensor_t** new_nodes = (nt_tensor_t**)realloc(g->nodes, 
                                                          new_cap * sizeof(nt_tensor_t*));
        if (!new_nodes) return (uint32_t)-1;
        g->nodes = new_nodes;
        g->nodes_capacity = new_cap;
    }
    
    uint32_t idx = g->n_nodes++;
    g->nodes[idx] = t;
    
    return idx;
}

nt_tensor_t* nt_graph_get_node(nt_graph_t* g, uint32_t index) {
    if (!g || index >= g->n_nodes) return NULL;
    return g->nodes[index];
}

int32_t nt_graph_find_node(nt_graph_t* g, const nt_tensor_t* t) {
    if (!g || !t) return -1;
    
    for (uint32_t i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i] == t) return (int32_t)i;
    }
    
    return -1;
}

/* ============================================================================
 * EDGE CONSTRUCTION
 * ============================================================================ */

nt_edge_t* nt_graph_add_edge(nt_graph_t* g, nt_edge_kind_t kind,
                              nt_tensor_t** inputs, uint16_t n_in,
                              nt_tensor_t** outputs, uint16_t n_out,
                              uint16_t op_code) {
    if (!g) return NULL;
    
    nt_edge_t* e = (nt_edge_t*)calloc(1, sizeof(nt_edge_t));
    if (!e) return NULL;
    
    e->kind = kind;
    e->edge_id = g->n_edges;
    e->op_code = op_code;
    e->state = NT_EDGE_PENDING;
    
    /* Copy inputs */
    if (n_in > 0 && inputs) {
        e->inputs = (nt_tensor_t**)malloc(n_in * sizeof(nt_tensor_t*));
        if (!e->inputs) {
            free(e);
            return NULL;
        }
        memcpy(e->inputs, inputs, n_in * sizeof(nt_tensor_t*));
        e->n_inputs = n_in;
    }
    
    /* Copy outputs */
    if (n_out > 0 && outputs) {
        e->outputs = (nt_tensor_t**)malloc(n_out * sizeof(nt_tensor_t*));
        if (!e->outputs) {
            if (e->inputs) free(e->inputs);
            free(e);
            return NULL;
        }
        memcpy(e->outputs, outputs, n_out * sizeof(nt_tensor_t*));
        e->n_outputs = n_out;
    }
    
    /* Add to linked list */
    e->prev = g->edges_tail;
    e->next = NULL;
    
    if (g->edges_tail) {
        g->edges_tail->next = e;
    } else {
        g->edges_head = e;
    }
    g->edges_tail = e;
    g->n_edges++;
    
    g->exec_order_valid = false;
    
    return e;
}

nt_edge_t* nt_graph_add_compute(nt_graph_t* g, uint16_t op,
                                 nt_tensor_t* dst,
                                 nt_tensor_t* src0,
                                 nt_tensor_t* src1) {
    nt_tensor_t* inputs[2] = {src0, src1};
    uint16_t n_in = src1 ? 2 : 1;
    
    return nt_graph_add_edge(g, NT_EDGE_COMPUTE, inputs, n_in, &dst, 1, op);
}

nt_edge_t* nt_graph_add_unary(nt_graph_t* g, uint16_t op,
                               nt_tensor_t* dst,
                               nt_tensor_t* src) {
    return nt_graph_add_edge(g, NT_EDGE_COMPUTE, &src, 1, &dst, 1, op);
}

void nt_edge_set_params(nt_edge_t* e, const int32_t* params, int n_params) {
    if (!e || !params) return;
    int n = n_params < 8 ? n_params : 8;
    memcpy(e->op_params, params, n * sizeof(int32_t));
}

void nt_edge_set_scalars(nt_edge_t* e, const float* scalars, int n_scalars) {
    if (!e || !scalars) return;
    int n = n_scalars < 4 ? n_scalars : 4;
    memcpy(e->op_scalars, scalars, n * sizeof(float));
}

void nt_edge_set_name(nt_edge_t* e, const char* name) {
    if (!e) return;
    e->name = name;
}

/* ============================================================================
 * SPECIAL HYPEREDGES
 * ============================================================================ */

nt_edge_t* nt_graph_add_attention(nt_graph_t* g,
                                   nt_tensor_t* Q,
                                   nt_tensor_t* K,
                                   nt_tensor_t* V,
                                   nt_tensor_t* output,
                                   nt_tensor_t* mask,
                                   int n_heads,
                                   float scale) {
    nt_tensor_t* inputs[4] = {Q, K, V, mask};
    uint16_t n_in = mask ? 4 : 3;
    
    nt_edge_t* e = nt_graph_add_edge(g, NT_EDGE_ATTENTION, inputs, n_in, &output, 1, 0);
    if (e) {
        e->op_params[0] = n_heads;
        e->op_scalars[0] = scale;
    }
    return e;
}

nt_edge_t* nt_graph_add_message(nt_graph_t* g,
                                 nt_tensor_t* node_features,
                                 nt_tensor_t* edge_index,
                                 nt_tensor_t* edge_attr,
                                 nt_tensor_t* output,
                                 int aggregation) {
    nt_tensor_t* inputs[3] = {node_features, edge_index, edge_attr};
    uint16_t n_in = edge_attr ? 3 : 2;
    
    nt_edge_t* e = nt_graph_add_edge(g, NT_EDGE_MESSAGE, inputs, n_in, &output, 1, 0);
    if (e) {
        e->op_params[0] = aggregation;
    }
    return e;
}

/* ============================================================================
 * GRAPH EXECUTION
 * ============================================================================ */

static void topological_sort_visit(nt_graph_t* g, nt_edge_t* e, 
                                   bool* visited, nt_edge_t** order, uint32_t* idx) {
    if (!e || visited[e->edge_id]) return;
    visited[e->edge_id] = true;
    
    /* Visit dependencies (edges that produce our inputs) */
    /* For simplicity, we just add in order here */
    
    order[(*idx)++] = e;
}

void nt_graph_build(nt_graph_t* g) {
    if (!g || g->exec_order_valid) return;
    
    /* Allocate execution order array */
    if (g->exec_order) free(g->exec_order);
    g->exec_order = (nt_edge_t**)malloc(g->n_edges * sizeof(nt_edge_t*));
    if (!g->exec_order) return;
    
    /* Simple topological sort */
    bool* visited = (bool*)calloc(g->n_edges, sizeof(bool));
    if (!visited) {
        free(g->exec_order);
        g->exec_order = NULL;
        return;
    }
    
    uint32_t idx = 0;
    nt_edge_t* e = g->edges_head;
    while (e) {
        topological_sort_visit(g, e, visited, g->exec_order, &idx);
        e = e->next;
    }
    
    g->exec_order_len = idx;
    g->exec_order_valid = true;
    
    free(visited);
}

void nt_graph_compute(nt_graph_t* g, int n_threads) {
    (void)n_threads;  /* Currently unused */
    if (!g) return;
    
    if (!g->exec_order_valid) {
        nt_graph_build(g);
    }
    
    /* Execute edges in order */
    for (uint32_t i = 0; i < g->exec_order_len; i++) {
        nt_edge_t* e = g->exec_order[i];
        if (!e) continue;
        
        e->state = NT_EDGE_RUNNING;
        
        /* Execute operation based on op_code */
        /* This would dispatch to actual compute kernels */
        
        e->state = NT_EDGE_COMPLETED;
    }
}

/* ============================================================================
 * PRINTING
 * ============================================================================ */

void nt_graph_print(const nt_graph_t* g) {
    if (!g) {
        printf("Graph: NULL\n");
        return;
    }
    
    printf("Graph: %s\n", g->name[0] ? g->name : "(unnamed)");
    printf("  Nodes: %u\n", g->n_nodes);
    printf("  Edges: %u\n", g->n_edges);
    printf("  Training: %s\n", g->is_training ? "yes" : "no");
    printf("  Grad enabled: %s\n", g->enable_grad ? "yes" : "no");
    
    if (g->n_nodes > 0) {
        printf("  Node list:\n");
        for (uint32_t i = 0; i < g->n_nodes && i < 10; i++) {
            nt_tensor_t* t = g->nodes[i];
            if (t) {
                const char* name = nt_tensor_get_name(t);
                printf("    [%u] %s: ", i, name ? name : "(unnamed)");
                printf("dtype=%s shape=[", nt_dtype_name(t->dtype));
                for (int d = 0; d < t->ndim; d++) {
                    printf("%d", t->ne[d]);
                    if (d < t->ndim - 1) printf(",");
                }
                printf("]\n");
            }
        }
        if (g->n_nodes > 10) {
            printf("    ... (%u more)\n", g->n_nodes - 10);
        }
    }
    
    if (g->n_edges > 0) {
        printf("  Edge list:\n");
        nt_edge_t* e = g->edges_head;
        int count = 0;
        while (e && count < 10) {
            printf("    [%lu] kind=%d inputs=%u outputs=%u\n",
                   e->edge_id, e->kind, e->n_inputs, e->n_outputs);
            e = e->next;
            count++;
        }
        if (g->n_edges > 10) {
            printf("    ... (%u more)\n", g->n_edges - 10);
        }
    }
}
