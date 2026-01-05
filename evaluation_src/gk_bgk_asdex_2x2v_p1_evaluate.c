#include <gkyl_null_comm.h>
#include <gkyl_comm_io.h>

#include <gkyl_gyrokinetic_priv.h>

int main(int argc, char **argv)
{
  printf("********************************\n");
  printf("BGK ASDEX Evaluation Benchmarks:\n");
  printf("********************************\n\n");

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

  struct gkyl_array *arr_training_final = gkyl_array_new(GKYL_DOUBLE, 4, ext_range.volume);
  struct gkyl_array *arr_validation_final = gkyl_array_new(GKYL_DOUBLE, 4, ext_range.volume);

  const char *fmt_training_final = "training_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-field_%d.gkyl";
  int sz_training_final = snprintf(0, 0, fmt_training_final, 99);
  char file_nm_training_final[sz_training_final + 1];
  snprintf(file_nm_training_final, sizeof file_nm_training_final, fmt_training_final, 99);

  int status_training_final = gkyl_comm_array_read(comm, &grid, &range, arr_training_final, file_nm_training_final);

  const char *fmt_validation_final = "validation_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-field_%d.gkyl";
  int sz_validation_final = snprintf(0, 0, fmt_validation_final, 99);
  char file_nm_validation_final[sz_validation_final + 1];
  snprintf(file_nm_validation_final, sizeof file_nm_validation_final, fmt_validation_final, 99);

  int status_validation_final = gkyl_comm_array_read(comm, &grid, &range, arr_validation_final, file_nm_validation_final);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &range);

  double l_infinity_final[4];
  double l_2_final[4];
  for (int i = 0; i < 4; i++) { 
    l_infinity_final[i] = 0.0;
    l_2_final[i] = 0.0;
  }

  long count = 0;
  while (gkyl_range_iter_next(&iter)) {
    long loc = gkyl_range_idx(&range, iter.idx);
    double *array_training_final = gkyl_array_fetch(arr_training_final, loc);
    double *array_validation_final = gkyl_array_fetch(arr_validation_final, loc);

    for (int i = 0; i < 4; i++) {
      if (fabs(array_training_final[i] - array_validation_final[i]) > l_infinity_final[i]) {
        l_infinity_final[i] = fabs(array_training_final[i] - array_validation_final[i]);
      }
      
      l_2_final[i] += fabs(array_training_final[i] - array_validation_final[i]) * fabs(array_training_final[i] - array_validation_final[i]);
    }
  }

  double l_infinity_final_all = 0.0;
  double l_2_final_all = 0.0;
  for (int i = 0; i < 4; i++) {
    if (l_infinity_final[i] > l_infinity_final_all) {
      l_infinity_final_all = l_infinity_final[i];
    }

    l_2_final_all += l_2_final[i];
    l_2_final[i] = sqrt(l_2_final[i]);
  }
  l_2_final_all = sqrt(l_2_final_all);

  printf("Final Frame Prediction:\n\n");

  for (int i = 0; i < 4; i++) {
    printf("  Criterion %d:\n", i + 1);
    printf("    L^infinity Error: %f\n", l_infinity_final[i]);
    printf("    L^2 Error: %f\n", l_2_final[i]);
  }
  printf("  Overall:\n");
  printf("    L^infinity Error: %f\n", l_infinity_final_all);
  printf("    L^2 Error: %f\n\n", l_2_final_all);

  double l_infinity_total[4];
  double l_2_total[4];
  for (int i = 0; i < 4; i++) { 
    l_infinity_total[i] = 0.0;
    l_2_total[i] = 0.0;
  }

  for (int i = 0; i < 100; i++) {
    struct gkyl_array *arr_training_total = gkyl_array_new(GKYL_DOUBLE, 4, ext_range.volume);
    struct gkyl_array *arr_validation_total = gkyl_array_new(GKYL_DOUBLE, 4, ext_range.volume);

    const char *fmt_training_total = "training_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-field_%d.gkyl";
    int sz_training_total = snprintf(0, 0, fmt_training_total, i);
    char file_nm_training_total[sz_training_total + 1];
    snprintf(file_nm_training_total, sizeof file_nm_training_total, fmt_training_total, i);

    int status_training_total = gkyl_comm_array_read(comm, &grid, &range, arr_training_total, file_nm_training_total);

    const char *fmt_validation_total = "validation_data/ASDEX_Gyrokinetic_P1/gk_bgk_asdex_2x2v_p1-field_%d.gkyl";
    int sz_validation_total = snprintf(0, 0, fmt_validation_total, i);
    char file_nm_validation_total[sz_validation_total + 1];
    snprintf(file_nm_validation_total, sizeof file_nm_validation_total, fmt_validation_total, i);

    int status_validation_total = gkyl_comm_array_read(comm, &grid, &range, arr_validation_total, file_nm_validation_total);

    struct gkyl_range_iter iter;
    gkyl_range_iter_init(&iter, &range);

    long count = 0;
    while (gkyl_range_iter_next(&iter)) {
      long loc = gkyl_range_idx(&range, iter.idx);
      double *array_training_total = gkyl_array_fetch(arr_training_total, loc);
      double *array_validation_total = gkyl_array_fetch(arr_validation_total, loc);

      for (int j = 0; j < 4; j++) {
        if (fabs(array_training_total[j] - array_validation_total[j]) > l_infinity_total[j]) {
          l_infinity_total[j] = fabs(array_training_total[j] - array_validation_total[j]);
        }
        
        l_2_total[j] += fabs(array_training_total[j] - array_validation_total[j]) * fabs(array_training_total[j] - array_validation_total[j]);
      }
    }
  }

  double l_infinity_total_all = 0.0;
  double l_2_total_all = 0.0;
  for (int i = 0; i < 4; i++) {
    if (l_infinity_total[i] > l_infinity_total_all) {
      l_infinity_total_all = l_infinity_total[i];
    }

    l_2_total_all += l_2_total[i];
    l_2_total[i] = sqrt(l_2_total[i]);
  }
  l_2_total_all = sqrt(l_2_total_all);

  printf("All Frame Prediction:\n\n");

  for (int i = 0; i < 4; i++) {
    printf("  Criterion %d:\n", i + 1);
    printf("    L^infinity Error: %f\n", l_infinity_total[i]);
    printf("    L^2 Error: %f\n", l_2_total[i]);
  }
  printf("  Overall:\n");
  printf("    L^infinity Error: %f\n", l_infinity_total_all);
  printf("    L^2 Error: %f\n", l_2_total_all);
}