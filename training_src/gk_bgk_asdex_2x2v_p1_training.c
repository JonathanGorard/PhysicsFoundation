#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <kann.h>

int main(int argc, char **argv)
{
  kad_node_t **t_net = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net[i] = kann_layer_dense(t_net[i], 256);
      t_net[i] = kad_tanh(t_net[i]);
    }

    t_net[i] = kann_layer_cost(t_net[i], 1, KANN_C_MSE);
    ann[i] = kann_new(t_net[i], 0);
  }

  float ***input_data = (float***) malloc(4 * sizeof(float**));
  float ***output_data = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
    output_data[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm;
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower[] = { 0.15, -M_PI + 1.0e-14 };
  double upper[] = { 0.172, M_PI - 1.0e-14 };
  int cells_new[] = { 128, 128 };
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, 2, lower, upper, cells_new);

  int nghost[] = { 2, 2 };
  struct gkyl_range range;
  struct gkyl_range ext_range;
  gkyl_create_grid_ranges(&grid, nghost, &ext_range, &range);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, 4, ext_range.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-field_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm, &grid, &range, arr, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr, loc);

      for (int j = 0; j < 4; j++) {
        input_data[j][(i * 128 * 128) + count] = (float*) malloc(3 * sizeof(float));
        output_data[j][(i * 128 * 128) + count] = (float*) malloc(sizeof(float));
      
        input_data[j][(i * 128 * 128) + count][0] = ((float)i) / 100.0;
        input_data[j][(i * 128 * 128) + count][1] = ((float)(count / 128)) / 128.0;
        input_data[j][(i * 128 * 128) + count][2] = ((float)(count % 128)) / 128.0;
        output_data[j][(i * 128 * 128) + count][0] = (float)c_array[j];
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann[i], 12, 100);
    kann_train_fnn1(ann[i], 0.0001f, 64, 50, 10, 0.1f, 128 * 128 * 100, input_data[i], output_data[i]);

    const char *fmt = "model_weights/gk_bgk_asdex_2x2v_p1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann[i]);
  }
  free(ann);
  free(t_net);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 128 * 128 * 100; j++) {
      free(input_data[i][j]);
      free(output_data[i][j]);
    }

    free(input_data[i]);
    free(output_data[i]);
  }

  free(input_data);
  free(output_data);

  kad_node_t **t_net_elcM1 = (kad_node_t**) malloc(4 * sizeof(kad_node_t*));
  kann_t **ann_elcM1 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) {
    t_net_elcM1[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net_elcM1[i] = kann_layer_dense(t_net_elcM1[i], 256);
      t_net_elcM1[i] = kad_tanh(t_net_elcM1[i]);
    }

    t_net_elcM1[i] = kann_layer_cost(t_net_elcM1[i], 1, KANN_C_MSE);
    ann_elcM1[i] = kann_new(t_net_elcM1[i], 0);
  }

  float ***input_data_elcM1 = (float***) malloc(4 * sizeof(float**));
  float ***output_data_elcM1 = (float***) malloc(4 * sizeof(float**));
  
  for (int i = 0; i < 4; i++) {
    input_data_elcM1[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
    output_data_elcM1[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm_elcM1;
  comm_elcM1 = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower_elcM1[] = { 0.15, -M_PI + 1.0e-14 };
  double upper_elcM1[] = { 0.172, M_PI - 1.0e-14 };
  int cells_new_elcM1[] = { 128, 128 };
  struct gkyl_rect_grid grid_elcM1;
  gkyl_rect_grid_init(&grid_elcM1, 2, lower_elcM1, upper_elcM1, cells_new_elcM1);

  int nghost_elcM1[] = { 2, 2 };
  struct gkyl_range range_elcM1;
  struct gkyl_range ext_range_elcM1;
  gkyl_create_grid_ranges(&grid_elcM1, nghost_elcM1, &ext_range_elcM1, &range_elcM1);

  struct gkyl_array *arr_elcM1 = gkyl_array_new(GKYL_DOUBLE, 4, ext_range_elcM1.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-elc_M1_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm_elcM1, &grid_elcM1, &range_elcM1, arr_elcM1, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elcM1);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elcM1, iter.idx);
      const double *c_array = gkyl_array_cfetch(arr_elcM1, loc);

      for (int j = 0; j < 4; j++) {
        input_data_elcM1[j][(i * 128 * 128) + count] = (float*) malloc(3 * sizeof(float));
        output_data_elcM1[j][(i * 128 * 128) + count] = (float*) malloc(sizeof(float));
      
        input_data_elcM1[j][(i * 128 * 128) + count][0] = ((float)i) / 100.0;
        input_data_elcM1[j][(i * 128 * 128) + count][1] = ((float)(count / 128)) / 128.0;
        input_data_elcM1[j][(i * 128 * 128) + count][2] = ((float)(count % 128)) / 128.0;
        output_data_elcM1[j][(i * 128 * 128) + count][0] = ((float)c_array[j]) / pow(10.0, 22.0);
      }

      count += 1;
    }
  }

  for (int i = 0; i < 4; i++) {
    kann_mt(ann_elcM1[i], 12, 100);
    kann_train_fnn1(ann_elcM1[i], 0.0001f, 64, 50, 10, 0.1f, 128 * 128 * 100, input_data_elcM1[i], output_data_elcM1[i]);

    const char *fmt = "model_weights/gk_bgk_asdex_2x2v_p1_elcM1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann_elcM1[i]);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elcM1[i]);
  }
  free(ann_elcM1);
  free(t_net_elcM1);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 128 * 128 * 100; j++) {
      free(input_data_elcM1[i][j]);
      free(output_data_elcM1[i][j]);
    }

    free(input_data_elcM1[i]);
    free(output_data_elcM1[i]);
  }

  free(input_data_elcM1);
  free(output_data_elcM1);
}