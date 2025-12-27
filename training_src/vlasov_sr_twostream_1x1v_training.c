#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <kann.h>

int main(int argc, char **argv)
{
  kad_node_t **t_net = (kad_node_t**) malloc(8 * sizeof(kad_node_t*));
  kann_t **ann = (kann_t**) malloc(8 * sizeof(kann_t*));

  for (int i = 0; i < 8; i++) {
    t_net[i] = kann_layer_input(3);
  
    for (int j = 0; j < 8; j++) {
      t_net[i] = kann_layer_dense(t_net[i], 256);
      t_net[i] = kad_tanh(t_net[i]);
    }

    t_net[i] = kann_layer_cost(t_net[i], 1, KANN_C_MSE);
    ann[i] = kann_new(t_net[i], 0);
  }

  float ***input_data = (float***) malloc(8 * sizeof(float**));
  float ***output_data = (float***) malloc(8 * sizeof(float**));
  
  for (int i = 0; i < 8; i++) {
    input_data[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
    output_data[i] = (float**) malloc(128 * 128 * 100 * sizeof(float*));
  }

  struct gkyl_comm *comm;
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower[] = { -2.0 * M_PI, -8.0 };
  double upper[] = { 2.0 * M_PI, 8.0 };
  int cells_new[] = { 128, 128 };
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, 2, lower, upper, cells_new);

  int nghost[] = { 2, 2 };
  struct gkyl_range range;
  struct gkyl_range ext_range;
  gkyl_create_grid_ranges(&grid, nghost, &ext_range, &range);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, 8, ext_range.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/TwoStream_SR_Vlasov_P2/vlasov_sr_twostream_1x1v-elc_%d.gkyl";
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

      for (int j = 0; j < 8; j++) {
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

  for (int i = 0; i < 8; i++) {
    kann_mt(ann[i], 12, 100);
    kann_train_fnn1(ann[i], 0.0001f, 64, 50, 10, 0.1f, 128 * 128 * 100, input_data[i], output_data[i]);

    const char *fmt = "model_weights/twostream_sr_vlasov_p2_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);
  
    kann_save(file_nm, ann[i]);
  }

  for (int i = 0; i < 8; i++) {
    kann_delete(ann[i]);
  }
  free(ann);
  free(t_net);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 128 * 128 * 100; j++) {
      free(input_data[i][j]);
      free(output_data[i][j]);
    }

    free(input_data[i]);
    free(output_data[i]);
  }

  free(input_data);
  free(output_data);
}