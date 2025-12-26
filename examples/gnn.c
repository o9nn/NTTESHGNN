/**
 * @file gnn.c
 * @brief NTTESHGNN - Graph Neural Network example
 * 
 * This example demonstrates computation graph construction and
 * basic GNN-style message passing operations.
 */

#include <stdio.h>
#include <math.h>
#include "ntteshgnn/ntteshgnn.h"

int main(void) {
    printf("NTTESHGNN Graph Neural Network Example\n");
    printf("======================================\n\n");
    
    /* Initialize library */
    nt_init();
    nt_context_t* ctx = nt_context_new();
    
    /* ========================================
     * 1. Create a computation graph
     * ======================================== */
    printf("1. Creating computation graph...\n");
    
    nt_graph_t* g = nt_graph_new(ctx);
    nt_graph_set_name(g, "simple_gnn");
    
    /* Create node feature tensors */
    int n_nodes = 6;
    int hidden_dim = 4;
    uint64_t seed = 12345;
    
    nt_tensor_t* node_features = nt_tensor_new_2d(ctx, NT_F32, hidden_dim, n_nodes);
    nt_tensor_rand(node_features, &seed);
    nt_tensor_set_name(node_features, "node_features");
    
    /* Create edge index [2, E] for edges:
     *     0 --- 1 --- 2
     *     |     |     |
     *     3 --- 4 --- 5
     */
    int n_edges = 14;  /* 7 undirected edges = 14 directed */
    nt_tensor_t* edge_index = nt_tensor_new_2d(ctx, NT_I32, n_edges, 2);
    int32_t* ei = (int32_t*)edge_index->data;
    
    /* Edge list (both directions for undirected) */
    int edges[][2] = {
        {0, 1}, {1, 0}, {1, 2}, {2, 1},
        {0, 3}, {3, 0}, {1, 4}, {4, 1},
        {2, 5}, {5, 2}, {3, 4}, {4, 3},
        {4, 5}, {5, 4}
    };
    
    for (int i = 0; i < n_edges; i++) {
        ei[i * 2 + 0] = edges[i][0];  /* source */
        ei[i * 2 + 1] = edges[i][1];  /* target */
    }
    nt_tensor_set_name(edge_index, "edge_index");
    
    /* Add to graph */
    nt_graph_add_node(g, node_features);
    nt_graph_add_node(g, edge_index);
    
    printf("  Created graph with %d nodes and %d edges\n", n_nodes, n_edges / 2);
    
    /* ========================================
     * 2. Print initial features
     * ======================================== */
    printf("\n2. Initial node features:\n");
    float* feat_data = (float*)node_features->data;
    for (int i = 0; i < n_nodes; i++) {
        printf("  Node %d: [", i);
        for (int d = 0; d < hidden_dim; d++) {
            printf("%.3f", feat_data[i * hidden_dim + d]);
            if (d < hidden_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    
    /* ========================================
     * 3. Compute adjacency matrix
     * ======================================== */
    printf("\n3. Adjacency matrix:\n");
    
    /* Build adjacency matrix from edge index */
    nt_tensor_t* adj = nt_tensor_new_2d(ctx, NT_F32, n_nodes, n_nodes);
    nt_tensor_zero(adj);
    float* A = (float*)adj->data;
    
    for (int e = 0; e < n_edges; e++) {
        int src = ei[e * 2 + 0];
        int dst = ei[e * 2 + 1];
        A[src * n_nodes + dst] = 1.0f;
    }
    
    printf("     ");
    for (int j = 0; j < n_nodes; j++) {
        printf("%4d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < n_nodes; i++) {
        printf("  %d: ", i);
        for (int j = 0; j < n_nodes; j++) {
            printf("%4.1f ", A[i * n_nodes + j]);
        }
        printf("\n");
    }
    
    /* ========================================
     * 4. Perform message passing (GCN-style)
     * ======================================== */
    printf("\n4. Message passing (GCN-style)...\n");
    
    /* Compute degree for normalization */
    float* degree = (float*)calloc(n_nodes, sizeof(float));
    for (int i = 0; i < n_nodes; i++) {
        degree[i] = 1.0f;  /* Self-loop */
        for (int j = 0; j < n_nodes; j++) {
            degree[i] += A[i * n_nodes + j];
        }
    }
    
    /* Add self-loops and normalize: D^{-1/2} (A + I) D^{-1/2} */
    nt_tensor_t* adj_norm = nt_tensor_new_2d(ctx, NT_F32, n_nodes, n_nodes);
    float* A_norm = (float*)adj_norm->data;
    
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_nodes; j++) {
            float a_ij = (i == j) ? 1.0f : A[i * n_nodes + j];
            A_norm[i * n_nodes + j] = a_ij / sqrtf(degree[i] * degree[j]);
        }
    }
    
    /* Message passing: H' = A_norm @ H */
    nt_tensor_t* new_features = nt_tensor_new_2d(ctx, NT_F32, hidden_dim, n_nodes);
    float* new_feat = (float*)new_features->data;
    
    for (int i = 0; i < n_nodes; i++) {
        for (int d = 0; d < hidden_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < n_nodes; j++) {
                sum += A_norm[i * n_nodes + j] * feat_data[j * hidden_dim + d];
            }
            /* Apply ReLU */
            new_feat[i * hidden_dim + d] = sum > 0.0f ? sum : 0.0f;
        }
    }
    
    printf("  Node features after message passing:\n");
    for (int i = 0; i < n_nodes; i++) {
        printf("    Node %d: [", i);
        for (int d = 0; d < hidden_dim; d++) {
            printf("%.3f", new_feat[i * hidden_dim + d]);
            if (d < hidden_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    
    /* ========================================
     * 5. Add message passing edge to graph
     * ======================================== */
    printf("\n5. Adding message passing edge to computation graph...\n");
    
    nt_graph_add_node(g, adj_norm);
    nt_graph_add_node(g, new_features);
    
    /* Add message passing hyperedge */
    nt_edge_t* mp_edge = nt_graph_add_message(g, node_features, edge_index, NULL, 
                                               new_features, 0);  /* 0 = sum aggregation */
    if (mp_edge) {
        nt_edge_set_name(mp_edge, "gcn_layer_1");
        printf("  Added message passing edge: %s\n", mp_edge->name);
    }
    
    /* ========================================
     * 6. Second layer of message passing
     * ======================================== */
    printf("\n6. Second message passing layer...\n");
    
    nt_tensor_t* output_features = nt_tensor_new_2d(ctx, NT_F32, hidden_dim, n_nodes);
    float* out_feat = (float*)output_features->data;
    
    for (int i = 0; i < n_nodes; i++) {
        for (int d = 0; d < hidden_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < n_nodes; j++) {
                sum += A_norm[i * n_nodes + j] * new_feat[j * hidden_dim + d];
            }
            /* Apply ReLU */
            out_feat[i * hidden_dim + d] = sum > 0.0f ? sum : 0.0f;
        }
    }
    
    printf("  Final node features:\n");
    for (int i = 0; i < n_nodes; i++) {
        printf("    Node %d: [", i);
        for (int d = 0; d < hidden_dim; d++) {
            printf("%.3f", out_feat[i * hidden_dim + d]);
            if (d < hidden_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    
    /* ========================================
     * 7. Graph readout (mean pooling)
     * ======================================== */
    printf("\n7. Graph readout (mean pooling)...\n");
    
    float graph_embedding[4] = {0};
    for (int i = 0; i < n_nodes; i++) {
        for (int d = 0; d < hidden_dim; d++) {
            graph_embedding[d] += out_feat[i * hidden_dim + d];
        }
    }
    for (int d = 0; d < hidden_dim; d++) {
        graph_embedding[d] /= n_nodes;
    }
    
    printf("  Graph embedding: [");
    for (int d = 0; d < hidden_dim; d++) {
        printf("%.3f", graph_embedding[d]);
        if (d < hidden_dim - 1) printf(", ");
    }
    printf("]\n");
    
    /* ========================================
     * 8. Print graph summary
     * ======================================== */
    printf("\n8. Computation graph summary:\n");
    nt_graph_print(g);
    
    /* ========================================
     * Cleanup
     * ======================================== */
    printf("\nCleaning up...\n");
    
    free(degree);
    nt_tensor_release(node_features);
    nt_tensor_release(edge_index);
    nt_tensor_release(adj);
    nt_tensor_release(adj_norm);
    nt_tensor_release(new_features);
    nt_tensor_release(output_features);
    nt_graph_free(g);
    nt_context_free(ctx);
    nt_cleanup();
    
    printf("Done!\n");
    return 0;
}
