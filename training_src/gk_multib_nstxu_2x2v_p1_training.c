#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <kann.h>

int main(int argc, char **argv)
{
  kad_node_t **t_net_block0 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block0 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block0[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block0[i] = kann_layer_dense(t_net_block0[i], 256);
      t_net_block0[i] = kad_tanh(t_net_block0[i]);
    }

    t_net_block0[i] = kann_layer_cost(t_net_block0[i], 1, KANN_C_MSE);
    ann_block0[i] = kann_new(t_net_block0[i], 0);
  }

  float ***input_data_block0 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block0 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block0[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_block0[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block0;
  comm_block0 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block0[] = { -3.717531 * pow(10.0, -2.0), -3.141593 };
  double upper_block0[] = { -3.544025 * pow(10.0, -2.0), 6.260430 * pow(10.0, -1.0) };
  int cells_new_block0[] = { 32, 24 };
  struct gkyl_rect_grid grid_block0;
  gkyl_rect_grid_init(&grid_block0, 2, lower_block0, upper_block0, cells_new_block0);

  int nghost_block0[] = { 2, 2 };
  struct gkyl_range range_block0;
  struct gkyl_range ext_range_block0;
  gkyl_create_grid_ranges(&grid_block0, nghost_block0, &ext_range_block0, &range_block0);

  struct gkyl_array *arr_block0 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block0.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b0-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block0, &grid_block0, &range_block0, arr_block0, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block0);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block0, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block0, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block0[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block0[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_block0[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_block0[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_block0[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_block0[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block0[i], 12, 100);
    kann_train_fnn1(ann_block0[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_block0[i], output_data_block0[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b0_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block0[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block0[i]);
  }
  free(ann_block0);
  free(t_net_block0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_block0[i][j]);
      free(output_data_block0[i][j]);
    }

    free(input_data_block0[i]);
    free(output_data_block0[i]);
  }

  free(input_data_block0);
  free(output_data_block0);

  kad_node_t **t_net_block1 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block1 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block1[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block1[i] = kann_layer_dense(t_net_block1[i], 256);
      t_net_block1[i] = kad_tanh(t_net_block1[i]);
    }

    t_net_block1[i] = kann_layer_cost(t_net_block1[i], 1, KANN_C_MSE);
    ann_block1[i] = kann_new(t_net_block1[i], 0);
  }

  float ***input_data_block1 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block1 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block1[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_block1[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block1;
  comm_block1 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block1[] = { -3.544025 * pow(10.0, -2.0), -3.141593 };
  double upper_block1[] = { -3.370519 * pow(10.0, -2.0), -2.271087 };
  int cells_new_block1[] = { 32, 24 };
  struct gkyl_rect_grid grid_block1;
  gkyl_rect_grid_init(&grid_block1, 2, lower_block1, upper_block1, cells_new_block1);

  int nghost_block1[] = { 2, 2 };
  struct gkyl_range range_block1;
  struct gkyl_range ext_range_block1;
  gkyl_create_grid_ranges(&grid_block1, nghost_block1, &ext_range_block1, &range_block1);

  struct gkyl_array *arr_block1 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block1.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b1-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block1, &grid_block1, &range_block1, arr_block1, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block1);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block1, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block1, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block1[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block1[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_block1[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_block1[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_block1[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_block1[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block1[i], 12, 100);
    kann_train_fnn1(ann_block1[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_block1[i], output_data_block1[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block1[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block1[i]);
  }
  free(ann_block1);
  free(t_net_block1);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_block1[i][j]);
      free(output_data_block1[i][j]);
    }

    free(input_data_block1[i]);
    free(output_data_block1[i]);
  }

  free(input_data_block1);
  free(output_data_block1);

  kad_node_t **t_net_block2 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block2 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block2[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block2[i] = kann_layer_dense(t_net_block2[i], 256);
      t_net_block2[i] = kad_tanh(t_net_block2[i]);
    }

    t_net_block2[i] = kann_layer_cost(t_net_block2[i], 1, KANN_C_MSE);
    ann_block2[i] = kann_new(t_net_block2[i], 0);
  }

  float ***input_data_block2 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block2 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block2[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_block2[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block2;
  comm_block2 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block2[] = { -3.544025 * pow(10.0, -2.0), -2.271087};
  double upper_block2[] = { -3.370519 * pow(10.0, -2.0), 0.0 };
  int cells_new_block2[] = { 32, 48 };
  struct gkyl_rect_grid grid_block2;
  gkyl_rect_grid_init(&grid_block2, 2, lower_block2, upper_block2, cells_new_block2);

  int nghost_block2[] = { 2, 2 };
  struct gkyl_range range_block2;
  struct gkyl_range ext_range_block2;
  gkyl_create_grid_ranges(&grid_block2, nghost_block2, &ext_range_block2, &range_block2);

  struct gkyl_array *arr_block2 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block2.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b2-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block2, &grid_block2, &range_block2, arr_block2, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block2);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block2, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block2, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block2[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block2[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_block2[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_block2[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_block2[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_block2[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block2[i], 12, 100);
    kann_train_fnn1(ann_block2[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_block2[i], output_data_block2[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b2_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block2[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block2[i]);
  }
  free(ann_block2);
  free(t_net_block2);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_block2[i][j]);
      free(output_data_block2[i][j]);
    }

    free(input_data_block2[i]);
    free(output_data_block2[i]);
  }

  free(input_data_block2);
  free(output_data_block2);

  kad_node_t **t_net_block3 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block3 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block3[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block3[i] = kann_layer_dense(t_net_block3[i], 256);
      t_net_block3[i] = kad_tanh(t_net_block3[i]);
    }

    t_net_block3[i] = kann_layer_cost(t_net_block3[i], 1, KANN_C_MSE);
    ann_block3[i] = kann_new(t_net_block3[i], 0);
  }

  float ***input_data_block3 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block3 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block3[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_block3[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block3;
  comm_block3 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block3[] = { -3.544025 * pow(10.0, -2.0), 0.0 };
  double upper_block3[] = { -3.370519 * pow(10.0, -2.0), 2.339689 };
  int cells_new_block3[] = { 32, 48 };
  struct gkyl_rect_grid grid_block3;
  gkyl_rect_grid_init(&grid_block3, 2, lower_block3, upper_block3, cells_new_block3);

  int nghost_block3[] = { 2, 2 };
  struct gkyl_range range_block3;
  struct gkyl_range ext_range_block3;
  gkyl_create_grid_ranges(&grid_block3, nghost_block3, &ext_range_block3, &range_block3);

  struct gkyl_array *arr_block3 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block3.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b3-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block3, &grid_block3, &range_block3, arr_block3, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block3);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block3, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block3, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block3[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block3[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_block3[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_block3[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_block3[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_block3[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block3[i], 12, 100);
    kann_train_fnn1(ann_block3[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_block3[i], output_data_block3[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b3_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block3[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block3[i]);
  }
  free(ann_block3);
  free(t_net_block3);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_block3[i][j]);
      free(output_data_block3[i][j]);
    }

    free(input_data_block3[i]);
    free(output_data_block3[i]);
  }

  free(input_data_block3);
  free(output_data_block3);

  kad_node_t **t_net_block4 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block4 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block4[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block4[i] = kann_layer_dense(t_net_block4[i], 256);
      t_net_block4[i] = kad_tanh(t_net_block4[i]);
    }

    t_net_block4[i] = kann_layer_cost(t_net_block4[i], 1, KANN_C_MSE);
    ann_block4[i] = kann_new(t_net_block4[i], 0);
  }

  float ***input_data_block4 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block4 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block4[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_block4[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block4;
  comm_block4 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block4[] = { -3.544025 * pow(10.0, -2.0), 2.339689 };
  double upper_block4[] = { -3.370519 * pow(10.0, -2.0), 3.141593 };
  int cells_new_block4[] = { 32, 24 };
  struct gkyl_rect_grid grid_block4;
  gkyl_rect_grid_init(&grid_block4, 2, lower_block4, upper_block4, cells_new_block4);

  int nghost_block4[] = { 2, 2 };
  struct gkyl_range range_block4;
  struct gkyl_range ext_range_block4;
  gkyl_create_grid_ranges(&grid_block4, nghost_block4, &ext_range_block4, &range_block4);

  struct gkyl_array *arr_block4 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block4.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b4-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block4, &grid_block4, &range_block4, arr_block4, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block4);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block4, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block4, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block4[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block4[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_block4[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_block4[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_block4[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_block4[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block4[i], 12, 100);
    kann_train_fnn1(ann_block4[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_block4[i], output_data_block4[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b4_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block4[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block4[i]);
  }
  free(ann_block4);
  free(t_net_block4);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_block4[i][j]);
      free(output_data_block4[i][j]);
    }

    free(input_data_block4[i]);
    free(output_data_block4[i]);
  }

  free(input_data_block4);
  free(output_data_block4);

  kad_node_t **t_net_block5 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block5 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block5[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block5[i] = kann_layer_dense(t_net_block5[i], 256);
      t_net_block5[i] = kad_tanh(t_net_block5[i]);
    }

    t_net_block5[i] = kann_layer_cost(t_net_block5[i], 1, KANN_C_MSE);
    ann_block5[i] = kann_new(t_net_block5[i], 0);
  }

  float ***input_data_block5 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block5 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block5[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_block5[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block5;
  comm_block5 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block5[] = { -3.717531 * pow(10.0, -2.0), 6.260430 * pow(10.0, -1.0)};
  double upper_block5[] = { -3.544025 * pow(10.0, -2.0), 3.141593 };
  int cells_new_block5[] = { 32, 24 };
  struct gkyl_rect_grid grid_block5;
  gkyl_rect_grid_init(&grid_block5, 2, lower_block5, upper_block5, cells_new_block5);

  int nghost_block5[] = { 2, 2 };
  struct gkyl_range range_block5;
  struct gkyl_range ext_range_block5;
  gkyl_create_grid_ranges(&grid_block5, nghost_block5, &ext_range_block5, &range_block5);

  struct gkyl_array *arr_block5 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block5.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b5-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block5, &grid_block5, &range_block5, arr_block5, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block5);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block5, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block5, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block5[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block5[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_block5[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_block5[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_block5[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_block5[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block5[i], 12, 100);
    kann_train_fnn1(ann_block5[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_block5[i], output_data_block5[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b5_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block5[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block5[i]);
  }
  free(ann_block5);
  free(t_net_block5);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_block5[i][j]);
      free(output_data_block5[i][j]);
    }

    free(input_data_block5[i]);
    free(output_data_block5[i]);
  }

  free(input_data_block5);
  free(output_data_block5);

  kad_node_t **t_net_block6 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block6 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block6[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block6[i] = kann_layer_dense(t_net_block6[i], 256);
      t_net_block6[i] = kad_tanh(t_net_block6[i]);
    }

    t_net_block6[i] = kann_layer_cost(t_net_block6[i], 1, KANN_C_MSE);
    ann_block6[i] = kann_new(t_net_block6[i], 0);
  }

  float ***input_data_block6 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block6 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block6[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_block6[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block6;
  comm_block6 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block6[] = { -3.717531 * pow(10.0, -2.0), -3.141593 };
  double upper_block6[] = { -3.544025 * pow(10.0, -2.0), -1.342996 };
  int cells_new_block6[] = { 32, 48 };
  struct gkyl_rect_grid grid_block6;
  gkyl_rect_grid_init(&grid_block6, 2, lower_block6, upper_block6, cells_new_block6);

  int nghost_block6[] = { 2, 2 };
  struct gkyl_range range_block6;
  struct gkyl_range ext_range_block6;
  gkyl_create_grid_ranges(&grid_block6, nghost_block6, &ext_range_block6, &range_block6);

  struct gkyl_array *arr_block6 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block6.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b6-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block6, &grid_block6, &range_block6, arr_block6, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block6);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block6, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block6, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block6[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block6[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_block6[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_block6[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_block6[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_block6[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block6[i], 12, 100);
    kann_train_fnn1(ann_block6[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_block6[i], output_data_block6[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b6_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block6[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block6[i]);
  }
  free(ann_block6);
  free(t_net_block6);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_block6[i][j]);
      free(output_data_block6[i][j]);
    }

    free(input_data_block6[i]);
    free(output_data_block6[i]);
  }

  free(input_data_block6);
  free(output_data_block6);

  kad_node_t **t_net_block7 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_block7 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_block7[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_block7[i] = kann_layer_dense(t_net_block7[i], 256);
      t_net_block7[i] = kad_tanh(t_net_block7[i]);
    }

    t_net_block7[i] = kann_layer_cost(t_net_block7[i], 1, KANN_C_MSE);
    ann_block7[i] = kann_new(t_net_block7[i], 0);
  }

  float ***input_data_block7 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_block7 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_block7[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_block7[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_block7;
  comm_block7 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_block7[] = { -3.717531 * pow(10.0, -2.0), 1.798597 };
  double upper_block7[] = { -3.544025 * pow(10.0, -2.0), 3.141593 };
  int cells_new_block7[] = { 32, 48 };
  struct gkyl_rect_grid grid_block7;
  gkyl_rect_grid_init(&grid_block7, 2, lower_block7, upper_block7, cells_new_block7);

  int nghost_block7[] = { 2, 2 };
  struct gkyl_range range_block7;
  struct gkyl_range ext_range_block7;
  gkyl_create_grid_ranges(&grid_block7, nghost_block7, &ext_range_block7, &range_block7);

  struct gkyl_array *arr_block7 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_block7.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b7-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_block7, &grid_block7, &range_block7, arr_block7, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block7);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block7, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_block7, loc);

      for (int j = 0; j < 4; j++) {
        input_data_block7[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_block7[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_block7[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_block7[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_block7[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_block7[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / 900.0;
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_block7[i], 12, 100);
    kann_train_fnn1(ann_block7[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_block7[i], output_data_block7[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b7_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_block7[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block7[i]);
  }
  free(ann_block7);
  free(t_net_block7);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_block7[i][j]);
      free(output_data_block7[i][j]);
    }

    free(input_data_block7[i]);
    free(output_data_block7[i]);
  }

  free(input_data_block7);
  free(output_data_block7);

  kad_node_t **t_net_elc_M1_block0 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block0 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block0[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block0[i] = kann_layer_dense(t_net_elc_M1_block0[i], 256);
      t_net_elc_M1_block0[i] = kad_tanh(t_net_elc_M1_block0[i]);
    }

    t_net_elc_M1_block0[i] = kann_layer_cost(t_net_elc_M1_block0[i], 1, KANN_C_MSE);
    ann_elc_M1_block0[i] = kann_new(t_net_elc_M1_block0[i], 0);
  }

  float ***input_data_elc_M1_block0 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block0 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block0[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_elc_M1_block0[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block0;
  comm_elc_M1_block0 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block0[] = { -3.717531 * pow(10.0, -2.0), -3.141593 };
  double upper_elc_M1_block0[] = { -3.544025 * pow(10.0, -2.0), 6.260430 * pow(10.0, -1.0) };
  int cells_new_elc_M1_block0[] = { 32, 24 };
  struct gkyl_rect_grid grid_elc_M1_block0;
  gkyl_rect_grid_init(&grid_elc_M1_block0, 2, lower_elc_M1_block0, upper_elc_M1_block0, cells_new_elc_M1_block0);

  int nghost_elc_M1_block0[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block0;
  struct gkyl_range ext_range_elc_M1_block0;
  gkyl_create_grid_ranges(&grid_elc_M1_block0, nghost_elc_M1_block0, &ext_range_elc_M1_block0, &range_elc_M1_block0);

  struct gkyl_array *arr_elc_M1_block0 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block0.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b0-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block0, &grid_elc_M1_block0, &range_elc_M1_block0, arr_elc_M1_block0, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block0);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block0, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block0, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block0[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block0[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block0[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block0[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_elc_M1_block0[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_elc_M1_block0[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block0[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block0[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_elc_M1_block0[i], output_data_elc_M1_block0[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b0_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block0[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block0[i]);
  }
  free(ann_elc_M1_block0);
  free(t_net_elc_M1_block0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_elc_M1_block0[i][j]);
      free(output_data_elc_M1_block0[i][j]);
    }

    free(input_data_elc_M1_block0[i]);
    free(output_data_elc_M1_block0[i]);
  }

  free(input_data_elc_M1_block0);
  free(output_data_elc_M1_block0);

  kad_node_t **t_net_elc_M1_block1 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block1 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block1[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block1[i] = kann_layer_dense(t_net_elc_M1_block1[i], 256);
      t_net_elc_M1_block1[i] = kad_tanh(t_net_elc_M1_block1[i]);
    }

    t_net_elc_M1_block1[i] = kann_layer_cost(t_net_elc_M1_block1[i], 1, KANN_C_MSE);
    ann_elc_M1_block1[i] = kann_new(t_net_elc_M1_block1[i], 0);
  }

  float ***input_data_elc_M1_block1 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block1 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block1[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_elc_M1_block1[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block1;
  comm_elc_M1_block1 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block1[] = { -3.544025 * pow(10.0, -2.0), -3.141593 };
  double upper_elc_M1_block1[] = { -3.370519 * pow(10.0, -2.0), -2.271087 };
  int cells_new_elc_M1_block1[] = { 32, 24 };
  struct gkyl_rect_grid grid_elc_M1_block1;
  gkyl_rect_grid_init(&grid_elc_M1_block1, 2, lower_elc_M1_block1, upper_elc_M1_block1, cells_new_elc_M1_block1);

  int nghost_elc_M1_block1[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block1;
  struct gkyl_range ext_range_elc_M1_block1;
  gkyl_create_grid_ranges(&grid_elc_M1_block1, nghost_elc_M1_block1, &ext_range_elc_M1_block1, &range_elc_M1_block1);

  struct gkyl_array *arr_elc_M1_block1 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block1.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b1-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block1, &grid_elc_M1_block1, &range_elc_M1_block1, arr_elc_M1_block1, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block1);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block1, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block1, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block1[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block1[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block1[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block1[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_elc_M1_block1[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_elc_M1_block1[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block1[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block1[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_elc_M1_block1[i], output_data_elc_M1_block1[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block1[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block1[i]);
  }
  free(ann_elc_M1_block1);
  free(t_net_elc_M1_block1);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_elc_M1_block1[i][j]);
      free(output_data_elc_M1_block1[i][j]);
    }

    free(input_data_elc_M1_block1[i]);
    free(output_data_elc_M1_block1[i]);
  }

  free(input_data_elc_M1_block1);
  free(output_data_elc_M1_block1);

  kad_node_t **t_net_elc_M1_block2 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block2 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block2[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block2[i] = kann_layer_dense(t_net_elc_M1_block2[i], 256);
      t_net_elc_M1_block2[i] = kad_tanh(t_net_elc_M1_block2[i]);
    }

    t_net_elc_M1_block2[i] = kann_layer_cost(t_net_elc_M1_block2[i], 1, KANN_C_MSE);
    ann_elc_M1_block2[i] = kann_new(t_net_elc_M1_block2[i], 0);
  }

  float ***input_data_elc_M1_block2 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block2 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block2[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_elc_M1_block2[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block2;
  comm_elc_M1_block2 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block2[] = { -3.544025 * pow(10.0, -2.0), -2.271087};
  double upper_elc_M1_block2[] = { -3.370519 * pow(10.0, -2.0), 0.0 };
  int cells_new_elc_M1_block2[] = { 32, 48 };
  struct gkyl_rect_grid grid_elc_M1_block2;
  gkyl_rect_grid_init(&grid_elc_M1_block2, 2, lower_elc_M1_block2, upper_elc_M1_block2, cells_new_elc_M1_block2);

  int nghost_elc_M1_block2[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block2;
  struct gkyl_range ext_range_elc_M1_block2;
  gkyl_create_grid_ranges(&grid_elc_M1_block2, nghost_elc_M1_block2, &ext_range_elc_M1_block2, &range_elc_M1_block2);

  struct gkyl_array *arr_elc_M1_block2 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block2.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b2-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block2, &grid_elc_M1_block2, &range_elc_M1_block2, arr_elc_M1_block2, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block2);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block2, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block2, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block2[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block2[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block2[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block2[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_elc_M1_block2[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_elc_M1_block2[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block2[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block2[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_elc_M1_block2[i], output_data_elc_M1_block2[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b2_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block2[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block2[i]);
  }
  free(ann_elc_M1_block2);
  free(t_net_elc_M1_block2);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_elc_M1_block2[i][j]);
      free(output_data_elc_M1_block2[i][j]);
    }

    free(input_data_elc_M1_block2[i]);
    free(output_data_elc_M1_block2[i]);
  }

  free(input_data_elc_M1_block2);
  free(output_data_elc_M1_block2);

  kad_node_t **t_net_elc_M1_block3 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block3 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block3[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block3[i] = kann_layer_dense(t_net_elc_M1_block3[i], 256);
      t_net_elc_M1_block3[i] = kad_tanh(t_net_elc_M1_block3[i]);
    }

    t_net_elc_M1_block3[i] = kann_layer_cost(t_net_elc_M1_block3[i], 1, KANN_C_MSE);
    ann_elc_M1_block3[i] = kann_new(t_net_elc_M1_block3[i], 0);
  }

  float ***input_data_elc_M1_block3 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block3 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block3[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_elc_M1_block3[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block3;
  comm_elc_M1_block3 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block3[] = { -3.544025 * pow(10.0, -2.0), 0.0 };
  double upper_elc_M1_block3[] = { -3.370519 * pow(10.0, -2.0), 2.339689 };
  int cells_new_elc_M1_block3[] = { 32, 48 };
  struct gkyl_rect_grid grid_elc_M1_block3;
  gkyl_rect_grid_init(&grid_elc_M1_block3, 2, lower_elc_M1_block3, upper_elc_M1_block3, cells_new_elc_M1_block3);

  int nghost_elc_M1_block3[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block3;
  struct gkyl_range ext_range_elc_M1_block3;
  gkyl_create_grid_ranges(&grid_elc_M1_block3, nghost_elc_M1_block3, &ext_range_elc_M1_block3, &range_elc_M1_block3);

  struct gkyl_array *arr_elc_M1_block3 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block3.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b3-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block3, &grid_elc_M1_block3, &range_elc_M1_block3, arr_elc_M1_block3, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block3);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block3, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block3, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block3[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block3[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block3[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block3[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_elc_M1_block3[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_elc_M1_block3[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block3[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block3[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_elc_M1_block3[i], output_data_elc_M1_block3[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b3_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block3[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block3[i]);
  }
  free(ann_elc_M1_block3);
  free(t_net_elc_M1_block3);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_elc_M1_block3[i][j]);
      free(output_data_elc_M1_block3[i][j]);
    }

    free(input_data_elc_M1_block3[i]);
    free(output_data_elc_M1_block3[i]);
  }

  free(input_data_elc_M1_block3);
  free(output_data_elc_M1_block3);

  kad_node_t **t_net_elc_M1_block4 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block4 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block4[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block4[i] = kann_layer_dense(t_net_elc_M1_block4[i], 256);
      t_net_elc_M1_block4[i] = kad_tanh(t_net_elc_M1_block4[i]);
    }

    t_net_elc_M1_block4[i] = kann_layer_cost(t_net_elc_M1_block4[i], 1, KANN_C_MSE);
    ann_elc_M1_block4[i] = kann_new(t_net_elc_M1_block4[i], 0);
  }

  float ***input_data_elc_M1_block4 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block4 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block4[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_elc_M1_block4[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block4;
  comm_elc_M1_block4 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block4[] = { -3.544025 * pow(10.0, -2.0), 2.339689 };
  double upper_elc_M1_block4[] = { -3.370519 * pow(10.0, -2.0), 3.141593 };
  int cells_new_elc_M1_block4[] = { 32, 24 };
  struct gkyl_rect_grid grid_elc_M1_block4;
  gkyl_rect_grid_init(&grid_elc_M1_block4, 2, lower_elc_M1_block4, upper_elc_M1_block4, cells_new_elc_M1_block4);

  int nghost_elc_M1_block4[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block4;
  struct gkyl_range ext_range_elc_M1_block4;
  gkyl_create_grid_ranges(&grid_elc_M1_block4, nghost_elc_M1_block4, &ext_range_elc_M1_block4, &range_elc_M1_block4);

  struct gkyl_array *arr_elc_M1_block4 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block4.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b4-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block4, &grid_elc_M1_block4, &range_elc_M1_block4, arr_elc_M1_block4, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block4);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block4, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block4, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block4[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block4[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block4[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block4[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_elc_M1_block4[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_elc_M1_block4[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block4[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block4[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_elc_M1_block4[i], output_data_elc_M1_block4[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b4_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block4[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block4[i]);
  }
  free(ann_elc_M1_block4);
  free(t_net_elc_M1_block4);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_elc_M1_block4[i][j]);
      free(output_data_elc_M1_block4[i][j]);
    }

    free(input_data_elc_M1_block4[i]);
    free(output_data_elc_M1_block4[i]);
  }

  free(input_data_elc_M1_block4);
  free(output_data_elc_M1_block4);

  kad_node_t **t_net_elc_M1_block5 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block5 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block5[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block5[i] = kann_layer_dense(t_net_elc_M1_block5[i], 256);
      t_net_elc_M1_block5[i] = kad_tanh(t_net_elc_M1_block5[i]);
    }

    t_net_elc_M1_block5[i] = kann_layer_cost(t_net_elc_M1_block5[i], 1, KANN_C_MSE);
    ann_elc_M1_block5[i] = kann_new(t_net_elc_M1_block5[i], 0);
  }

  float ***input_data_elc_M1_block5 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block5 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block5[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
    output_data_elc_M1_block5[i] = (float**) malloc(32 * 24 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block5;
  comm_elc_M1_block5 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block5[] = { -3.717531 * pow(10.0, -2.0), 6.260430 * pow(10.0, -1.0)};
  double upper_elc_M1_block5[] = { -3.544025 * pow(10.0, -2.0), 3.141593 };
  int cells_new_elc_M1_block5[] = { 32, 24 };
  struct gkyl_rect_grid grid_elc_M1_block5;
  gkyl_rect_grid_init(&grid_elc_M1_block5, 2, lower_elc_M1_block5, upper_elc_M1_block5, cells_new_elc_M1_block5);

  int nghost_elc_M1_block5[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block5;
  struct gkyl_range ext_range_elc_M1_block5;
  gkyl_create_grid_ranges(&grid_elc_M1_block5, nghost_elc_M1_block5, &ext_range_elc_M1_block5, &range_elc_M1_block5);

  struct gkyl_array *arr_elc_M1_block5 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block5.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b5-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block5, &grid_elc_M1_block5, &range_elc_M1_block5, arr_elc_M1_block5, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block5);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block5, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block5, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block5[j][(i * 32 * 24) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block5[j][(i * 32 * 24) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block5[j][(i * 32 * 24) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block5[j][(i * 32 * 24) + count][1] = ((float)(count / 24)) / 32.0;
        input_data_elc_M1_block5[j][(i * 32 * 24) + count][2] = ((float)(count % 24)) / 24.0;
        output_data_elc_M1_block5[j][(i * 32 * 24) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block5[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block5[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 24 * 100, input_data_elc_M1_block5[i], output_data_elc_M1_block5[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b5_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block5[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block5[i]);
  }
  free(ann_elc_M1_block5);
  free(t_net_elc_M1_block5);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 24 * 100; j++) {
      free(input_data_elc_M1_block5[i][j]);
      free(output_data_elc_M1_block5[i][j]);
    }

    free(input_data_elc_M1_block5[i]);
    free(output_data_elc_M1_block5[i]);
  }

  free(input_data_elc_M1_block5);
  free(output_data_elc_M1_block5);

  kad_node_t **t_net_elc_M1_block6 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block6 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block6[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block6[i] = kann_layer_dense(t_net_elc_M1_block6[i], 256);
      t_net_elc_M1_block6[i] = kad_tanh(t_net_elc_M1_block6[i]);
    }

    t_net_elc_M1_block6[i] = kann_layer_cost(t_net_elc_M1_block6[i], 1, KANN_C_MSE);
    ann_elc_M1_block6[i] = kann_new(t_net_elc_M1_block6[i], 0);
  }

  float ***input_data_elc_M1_block6 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block6 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block6[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_elc_M1_block6[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block6;
  comm_elc_M1_block6 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block6[] = { -3.717531 * pow(10.0, -2.0), -3.141593 };
  double upper_elc_M1_block6[] = { -3.544025 * pow(10.0, -2.0), -1.342996 };
  int cells_new_elc_M1_block6[] = { 32, 48 };
  struct gkyl_rect_grid grid_elc_M1_block6;
  gkyl_rect_grid_init(&grid_elc_M1_block6, 2, lower_elc_M1_block6, upper_elc_M1_block6, cells_new_elc_M1_block6);

  int nghost_elc_M1_block6[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block6;
  struct gkyl_range ext_range_elc_M1_block6;
  gkyl_create_grid_ranges(&grid_elc_M1_block6, nghost_elc_M1_block6, &ext_range_elc_M1_block6, &range_elc_M1_block6);

  struct gkyl_array *arr_elc_M1_block6 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block6.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b6-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block6, &grid_elc_M1_block6, &range_elc_M1_block6, arr_elc_M1_block6, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block6);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block6, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block6, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block6[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block6[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block6[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block6[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_elc_M1_block6[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_elc_M1_block6[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block6[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block6[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_elc_M1_block6[i], output_data_elc_M1_block6[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b6_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block6[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block6[i]);
  }
  free(ann_elc_M1_block6);
  free(t_net_elc_M1_block6);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_elc_M1_block6[i][j]);
      free(output_data_elc_M1_block6[i][j]);
    }

    free(input_data_elc_M1_block6[i]);
    free(output_data_elc_M1_block6[i]);
  }

  free(input_data_elc_M1_block6);
  free(output_data_elc_M1_block6);

  kad_node_t **t_net_elc_M1_block7 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elc_M1_block7 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elc_M1_block7[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elc_M1_block7[i] = kann_layer_dense(t_net_elc_M1_block7[i], 256);
      t_net_elc_M1_block7[i] = kad_tanh(t_net_elc_M1_block7[i]);
    }

    t_net_elc_M1_block7[i] = kann_layer_cost(t_net_elc_M1_block7[i], 1, KANN_C_MSE);
    ann_elc_M1_block7[i] = kann_new(t_net_elc_M1_block7[i], 0);
  }

  float ***input_data_elc_M1_block7 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elc_M1_block7 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elc_M1_block7[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
    output_data_elc_M1_block7[i] = (float**) malloc(32 * 48 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elc_M1_block7;
  comm_elc_M1_block7 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elc_M1_block7[] = { -3.717531 * pow(10.0, -2.0), 1.798597 };
  double upper_elc_M1_block7[] = { -3.544025 * pow(10.0, -2.0), 3.141593 };
  int cells_new_elc_M1_block7[] = { 32, 48 };
  struct gkyl_rect_grid grid_elc_M1_block7;
  gkyl_rect_grid_init(&grid_elc_M1_block7, 2, lower_elc_M1_block7, upper_elc_M1_block7, cells_new_elc_M1_block7);

  int nghost_elc_M1_block7[] = { 2, 2 };
  struct gkyl_range range_elc_M1_block7;
  struct gkyl_range ext_range_elc_M1_block7;
  gkyl_create_grid_ranges(&grid_elc_M1_block7, nghost_elc_M1_block7, &ext_range_elc_M1_block7, &range_elc_M1_block7);

  struct gkyl_array *arr_elc_M1_block7 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elc_M1_block7.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b7-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elc_M1_block7, &grid_elc_M1_block7, &range_elc_M1_block7, arr_elc_M1_block7, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block7);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block7, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elc_M1_block7, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elc_M1_block7[j][(i * 32 * 48) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elc_M1_block7[j][(i * 32 * 48) + count] = (float*) malloc(sizeof(float));
      
        input_data_elc_M1_block7[j][(i * 32 * 48) + count][0] = ((float)i) / 100.0;
        input_data_elc_M1_block7[j][(i * 32 * 48) + count][1] = ((float)(count / 48)) / 32.0;
        input_data_elc_M1_block7[j][(i * 32 * 48) + count][2] = ((float)(count % 48)) / 48.0;
        output_data_elc_M1_block7[j][(i * 32 * 48) + count][0] = ((float)c_array[j]) / pow(10.0, 26.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elc_M1_block7[i], 12, 100);
    kann_train_fnn1(ann_elc_M1_block7[i], 0.0001f, 64, 50, 10, 0.1f, 32 * 48 * 100, input_data_elc_M1_block7[i], output_data_elc_M1_block7[i]);

    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b7_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elc_M1_block7[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block7[i]);
  }
  free(ann_elc_M1_block7);
  free(t_net_elc_M1_block7);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 32 * 48 * 100; j++) {
      free(input_data_elc_M1_block7[i][j]);
      free(output_data_elc_M1_block7[i][j]);
    }

    free(input_data_elc_M1_block7[i]);
    free(output_data_elc_M1_block7[i]);
  }

  free(input_data_elc_M1_block7);
  free(output_data_elc_M1_block7);
}