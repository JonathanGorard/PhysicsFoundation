#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <gkyl_gyrokinetic_priv.h>
#include <kann.h>

int main(int argc, char **argv)
{
  kann_t **ann_block0 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b0_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block0[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block0, &grid_block0, &range_block0, arr_block0, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block0);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block0, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block0, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_block0[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b0-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block0, &grid_block0, &range_block0, mt, arr_block0, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block0[i]);
  }
  free(ann_block0);

  kann_t **ann_block1 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block1[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block1, &grid_block1, &range_block1, arr_block1, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block1);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block1, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block1, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_block1[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b1-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block1, &grid_block1, &range_block1, mt, arr_block1, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block1[i]);
  }
  free(ann_block1);

  kann_t **ann_block2 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b2_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block2[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block2, &grid_block2, &range_block2, arr_block2, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block2);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block2, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block2, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_block2[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b2-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block2, &grid_block2, &range_block2, mt, arr_block2, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block2[i]);
  }
  free(ann_block2);

  kann_t **ann_block3 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b3_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block3[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block3, &grid_block3, &range_block3, arr_block3, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block3);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block3, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block3, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_block3[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b3-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block3, &grid_block3, &range_block3, mt, arr_block3, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block3[i]);
  }
  free(ann_block3);

  kann_t **ann_block4 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b4_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block4[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block4, &grid_block4, &range_block4, arr_block4, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block4);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block4, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block4, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_block4[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b4-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block4, &grid_block4, &range_block4, mt, arr_block4, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block4[i]);
  }
  free(ann_block4);

  kann_t **ann_block5 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b5_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block5[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block5, &grid_block5, &range_block5, arr_block5, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block5);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block5, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block5, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_block5[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b5-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block5, &grid_block5, &range_block5, mt, arr_block5, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block5[i]);
  }
  free(ann_block5);

  kann_t **ann_block6 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b6_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block6[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block6, &grid_block6, &range_block6, arr_block6, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block6);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block6, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block6, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_block6[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b6-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block6, &grid_block6, &range_block6, mt, arr_block6, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block6[i]);
  }
  free(ann_block6);

  kann_t **ann_block7 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_b7_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_block7[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_block7, &grid_block7, &range_block7, arr_block7, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_block7);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_block7, iter.idx);
      double *array_new = gkyl_array_fetch(arr_block7, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_block7[j], input_data);

        array_new[j] = output_data[0] * 900.0;

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b7-field_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_block7, &grid_block7, &range_block7, mt, arr_block7, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_block7[i]);
  }
  free(ann_block7);

  kann_t **ann_elc_M1_block0 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b0_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block0[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block0, &grid_elc_M1_block0, &range_elc_M1_block0, arr_elc_M1_block0, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block0);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block0, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block0, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_elc_M1_block0[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b0-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block0, &grid_elc_M1_block0, &range_elc_M1_block0, mt, arr_elc_M1_block0, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block0[i]);
  }
  free(ann_elc_M1_block0);

  kann_t **ann_elc_M1_block1 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b1_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block1[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block1, &grid_elc_M1_block1, &range_elc_M1_block1, arr_elc_M1_block1, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block1);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block1, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block1, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_elc_M1_block1[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b1-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block1, &grid_elc_M1_block1, &range_elc_M1_block1, mt, arr_elc_M1_block1, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block1[i]);
  }
  free(ann_elc_M1_block1);

  kann_t **ann_elc_M1_block2 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b2_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block2[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block2, &grid_elc_M1_block2, &range_elc_M1_block2, arr_elc_M1_block2, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block2);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block2, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block2, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_elc_M1_block2[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b2-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block2, &grid_elc_M1_block2, &range_elc_M1_block2, mt, arr_elc_M1_block2, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block2[i]);
  }
  free(ann_elc_M1_block2);

  kann_t **ann_elc_M1_block3 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b3_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block3[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block3, &grid_elc_M1_block3, &range_elc_M1_block3, arr_elc_M1_block3, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block3);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block3, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block3, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_elc_M1_block3[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b3-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block3, &grid_elc_M1_block3, &range_elc_M1_block3, mt, arr_elc_M1_block3, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block3[i]);
  }
  free(ann_elc_M1_block3);

  kann_t **ann_elc_M1_block4 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b4_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block4[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block4, &grid_elc_M1_block4, &range_elc_M1_block4, arr_elc_M1_block4, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block4);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block4, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block4, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_elc_M1_block4[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b4-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block4, &grid_elc_M1_block4, &range_elc_M1_block4, mt, arr_elc_M1_block4, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block4[i]);
  }
  free(ann_elc_M1_block4);

  kann_t **ann_elc_M1_block5 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b5_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block5[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block5, &grid_elc_M1_block5, &range_elc_M1_block5, arr_elc_M1_block5, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block5);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block5, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block5, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 24)) / 32.0;
        input_data[2] = ((float)(count % 24)) / 24.0;
      
        output_data = kann_apply1(ann_elc_M1_block5[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b5-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block5, &grid_elc_M1_block5, &range_elc_M1_block5, mt, arr_elc_M1_block5, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block5[i]);
  }
  free(ann_elc_M1_block5);

  kann_t **ann_elc_M1_block6 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b6_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block6[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block6, &grid_elc_M1_block6, &range_elc_M1_block6, arr_elc_M1_block6, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block6);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block6, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block6, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_elc_M1_block6[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b6-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block6, &grid_elc_M1_block6, &range_elc_M1_block6, mt, arr_elc_M1_block6, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block6[i]);
  }
  free(ann_elc_M1_block6);

  kann_t **ann_elc_M1_block7 = (kann_t**) malloc(4 * sizeof(kann_t*));

  for (int i = 0; i < 4; i++) { 
    const char *fmt = "model_weights/gk_multib_nstxu_2x2v_p1_elcM1_b7_%d_neural_net.dat";
    int sz = snprintf(0, 0, fmt, i);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, i);

    FILE *fptr;
    fptr = fopen(file_nm, "r");
    if (fptr != NULL) {
      ann_elc_M1_block7[i] = kann_load(file_nm);
    
      fclose(fptr);
    }
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
    int sz = snprintf(0, 0, fmt, 0);
    char file_nm[sz + 1];
    snprintf(file_nm, sizeof file_nm, fmt, 0);

    int status = gkyl_comm_array_read(comm_elc_M1_block7, &grid_elc_M1_block7, &range_elc_M1_block7, arr_elc_M1_block7, file_nm);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range_elc_M1_block7);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range_elc_M1_block7, iter.idx);
      double *array_new = gkyl_array_fetch(arr_elc_M1_block7, loc);

      for (int j = 0; j < 4; j++) {
        float *input_data = (float*) malloc(3 * sizeof(float));
        const float *output_data;

        input_data[0] = ((float)i) / 100.0;
        input_data[1] = ((float)(count / 48)) / 32.0;
        input_data[2] = ((float)(count % 48)) / 48.0;
      
        output_data = kann_apply1(ann_elc_M1_block7[j], input_data);

        array_new[j] = output_data[0] * pow(10.0, 26.0);

        free(input_data);
      }

      count += 1;
    }

    const char *fmt_new = "validation_data/NSTXU_Gyrokinetic_P1/gk_multib_nstxu_2x2v_p1_b7-elc_M1_%d.gkyl";
    int sz_new = snprintf(0, 0, fmt_new, i);
    char file_nm_new[sz_new + 1];
    snprintf(file_nm_new, sizeof file_nm_new, fmt_new, i);

    struct gkyl_msgpack_data *mt = gk_array_meta_new( (struct gyrokinetic_output_meta) {
      .frame = i,
      .stime = (double)i,
      .poly_order = 1,
      .basis_type = "serendipity",
    }, GKYL_GK_META_NONE, 0
  );

    gkyl_comm_array_write(comm_elc_M1_block7, &grid_elc_M1_block7, &range_elc_M1_block7, mt, arr_elc_M1_block7, file_nm_new);
  }

  for (int i = 0; i < 4; i++) {
    kann_delete(ann_elc_M1_block7[i]);
  }
  free(ann_elc_M1_block7);
}