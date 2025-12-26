#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <gkyl_vlasov_priv.h>
#include <kann.h>

int main(int argc, char **argv)
{
  kann_t **ann = (kann_t**) malloc(2 * sizeof(kann_t*));

  for (int i = 0; i < 2; i++) { 
    const char *fmt = "model_weights/twostream_vlasov_p1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
  }

  struct gkyl_comm *comm;
  comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .use_gpu = false  
    }
  );

  double lower[] = { -10.0 * M_PI, -24.0 };
  double upper[] = { 10.0 * M_PI, 24.0 };
  int cells_new[] = { 128, 128 };
  struct gkyl_rect_grid grid;
  gkyl_rect_grid_init(&grid, 2, lower, upper, cells_new);

  int nghost[] = { 2, 2 };
  struct gkyl_range range;
  struct gkyl_range ext_range;
  gkyl_create_grid_ranges(&grid, nghost, &ext_range, &range);

  struct gkyl_array *arr = gkyl_array_new(GKYL_DOUBLE, 2, ext_range.volume);

  for (int i = 0; i < 100; i++) {
    const char *fmt = "training_data/TwoStream_Vlasov_P1/vlasov_twostream_p1-elc_%d.gkyl";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    int status = gkyl_comm_array_read(comm, &grid, &range, arr, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range, iter.idx);
      double *array_new = gkyl_array_fetch(arr, loc);

      for (int j = 0; j < 2; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 128)) / 128.0;
        input_data[2] = ((float)(count % 128)) / 128.0;
      
        output_data = kann_apply1(ann[j], input_data);

        array_new[j] = output_data[0];

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/TwoStream_Vlasov_P1/vlasov_twostream_p1-elc_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = vlasov_array_meta_new( (struct vlasov_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }
  );

    gkyl_comm_array_write(comm, &grid, &range, mt, arr, file_nm_new);
  }

  for (int i = 0; i < 2; i++) {
    kann_delete(ann[i]);
  }
  free(ann);
}