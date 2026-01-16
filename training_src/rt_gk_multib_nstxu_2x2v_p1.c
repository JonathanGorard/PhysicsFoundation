#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_efit.h>
#include <gkyl_gyrokinetic_multib.h>
#include <gkyl_mpi_comm.h>
#include <gkyl_null_comm.h>
#include <gkyl_tok_geo.h>

#include <rt_arg_parse.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#endif

//void shaped_pfunc_lower_outer(double s, double* RZ){
//  RZ[0] = 1.0+0.7*s;
//  RZ[1] = -2.07993;
//}
//
//void shaped_pfunc_upper_outer(double s, double* RZ){
//  RZ[0] = 1.0+0.7*s;
//  RZ[1] = 2.07993;
//}

//// SIMPLE HORIZONTAL AND VERTICAL PLATES

// void shaped_pfunc_lower_outer(double s, double* RZ){
//   RZ[0] = 0.3+0.7*s;
//   RZ[1] = -1.5;
// }

// void shaped_pfunc_upper_outer(double s, double* RZ){
//   RZ[0] = 0.3+0.7*s;
//   RZ[1] = 1.5;
// }

// void shaped_pfunc_upper_inner(double s, double* RZ){
//     RZ[0] = 0.3;
//     RZ[1] = 1.0+ (1.5 - 1.0)*s;
// }

// void shaped_pfunc_lower_inner(double s, double* RZ){
//     RZ[0] = 0.3;
//     RZ[1] = -(1.0 + (1.5 - 1.0)*s);
// }

//// ACTUAL SHAPED PLATES

void shaped_pfunc_lower_outer(double s, double* RZ){
  // Linear parametric segment between (r1,z1)=(0.55,-1.7) and (r2,z2)=(1.2,-1.5); s in [0,1]
  RZ[0] = 0.55 + (1.2 - 0.55)*s;     // 0.55 + 0.65*s
  RZ[1] = -1.7 + (-1.5 + 1.7)*s;     // -1.7 + 0.2*s
}

void shaped_pfunc_upper_outer(double s, double* RZ){
  // Symmetric (Z flipped) segment: (0.55,+1.7) to (1.2,+1.5); s in [0,1]
  RZ[0] = 0.55 + (1.2 - 0.55)*s;     // 0.55 + 0.65*s
  RZ[1] =  1.7 + (1.5 - 1.7)*s;      //  1.7 - 0.2*s
}

void shaped_pfunc_upper_inner(double s, double* RZ){
  // Upper inner plate: vertical flip of lower inner plate
  // (r1,z1) = (0.2, +1.2) -> (r2,z2) = (0.35, +1.4)
  RZ[0] = 0.2 + (0.35 - 0.2)*s;    // 0.2 + 0.15*s
  RZ[1] = 1.2 + (1.4 - 1.2)*s;     // 1.2 + 0.2*s
}

void shaped_pfunc_lower_inner(double s, double* RZ){
  // Lower inner plate: (r1,z1) = (0.2, -1.2) -> (r2,z2) = (0.35, -1.4)
  RZ[0] = 0.2 + (0.35 - 0.2)*s;    // 0.2 + 0.15*s
  RZ[1] = -1.2 + (-1.4 + 1.2)*s;   // -1.2 - 0.2*s
}


struct gkyl_gk_block_geom*
create_gk_block_geom(void)
{
  // Only do b1-3 in the block layout below.
  struct gkyl_gk_block_geom *bgeom = gkyl_gk_block_geom_new(2, 8);

  /* Block layout and coordinates

   x  
   ^  
   |
   4  +------------------+------------------+------------------+
   |  |b1                |b2                |b3                |
   |  |lower outer SOL   |middle outer sol  |upper outer sol   |
   |  |                  |                  |                  |
   3  +------------------+------------------+------------------+
   |  |b0               x|o b10            %|$ b4              |
   |  |lower outer PF   x|o outer core     %|$ upper outer PF  |
   |  |                 x|o                %|$                 |
   |  +------------------+------------------+------------------+
   2  +------------------+------------------+------------------+
   |  |b9               x|o b11            %|$ b5              |
   |  |lower inner PF   x|o inner core     %|$ upper inner PF  |
   |  |                 x|o                %|$                 |
   1  +------------------+------------------+------------------+
   |  |b8                |b7                |b6                |
   |  |lower inner SOL   |middle inner SOL  |upper inner SOL   |
   |  |                  |                  |                  |
   0  +------------------+------------------+------------------+

      0 -----------1------------2------------3 -> z

      Edges that touch coincide are physically connected unless
      otherwise indicated by a special symbol. Edges with a special
      symbol such as o,x,%, or % are instead connected to the other
      edge with the same symbol. Edges that do not coincide with
      another edge are a physical boundary.
  */  


  struct gkyl_efit_inp efit_inp = {
    .filepath = "/Users/jg5510/Documents/gkeyll_foundation_model/gyrokinetic/data/eqdsk/nstxu_DN.geqdsk",
    .rz_poly_order = 2,
    .flux_poly_order = 1,
    .reflect = true,
  };

  struct gkyl_efit *efit = gkyl_efit_new(&efit_inp);
  double psisep = efit->psisep;
  gkyl_efit_release(efit);
  //double psisep = -0.0354402478890806;

  // ** hard-coded conventions from h11 test from Akash **

  // double dsep = 0.000;

  // //double psi_up_outer_sol = -0.08;
  // double psi_up_outer_sol = -0.10289308777929265;
  // double psi_lo_outer_sol = psisep + dsep;

  // //double psi_lo_core = -0.17;
  // // For core equal cells as outboard : -0.17157235111717056
  // // For core 1/2 cells as outboard: -0.14867926333787793
  // // For core 1/3 cells as outboard: -0.14104823407811373
  // double psi_lo_core = -0.14867926333787793;
  // double psi_up_core = psisep - dsep;

  // //double psi_lo_pf = -0.14;
  // double psi_lo_pf = -0.1334172048183495;
  // double psi_up_pf = psisep - dsep;

  // //double psi_up_inner_sol = -0.11;
  // double psi_up_inner_sol = -0.11052411703905686;
  // double psi_lo_inner_sol = psisep + dsep;

  // width conventions from step10

  double dsep = 0.000;

  // Old calculated values for uniform width

  double wout = 0.0069402478890806/4.0;
  double win = 0.0069402478890806/4.0;
  double wcore = 0.0069402478890806/4.0;
  double wpf = 0.0069402478890806/4.0;


  // New values for hardcoded widths

  // double wout = 0.02;
  // double win = 0.0075;
  // double wcore = 0.02;
  // double wpf = 0.0075;


  double psi_lo_outer_sol = psisep;
  double psi_up_outer_sol = psisep + wout;

  double psi_lo_core = psisep - wcore;
  double psi_up_core = psisep;

  double psi_lo_pf = psisep - wpf;
  double psi_up_pf = psisep;

  double psi_lo_inner_sol = psisep;
  double psi_up_inner_sol = psisep + win;

  // OLD GRID

  int npsi_outer_sol = 64;
  int npsi_core = 64;
  int npsi_pf = 64;
  int npsi_inner_sol = 64;

  double ntheta_lower_inner  = 96/2;
  double ntheta_middle_inner = 96;
  double ntheta_upper_inner  = 96/2;

  double ntheta_lower_outer = 96/2;
  double ntheta_middle_outer = 96;
  double ntheta_upper_outer = 96/2;


  // H16 GRID

  // int npsi_outer_sol = 6;
  // int npsi_core = 6;
  // int npsi_pf = 2;
  // int npsi_inner_sol = 4;

  // double ntheta_lower_inner  = 4;
  // double ntheta_middle_inner = 8;
  // double ntheta_upper_inner  = 4;

  // double ntheta_lower_outer = 8;
  // double ntheta_middle_outer = 10;
  // double ntheta_upper_outer = 8;

  // COARSE H16 GRID

  // int npsi_outer_sol = 6/2;
  // int npsi_core = 6/2;
  // int npsi_pf = 2/2;
  // int npsi_inner_sol = 4/2;

  // double ntheta_lower_inner  = 4/2;
  // double ntheta_middle_inner = 8/2;
  // double ntheta_upper_inner  = 4/2;

  // double ntheta_lower_outer = 8/2;
  // double ntheta_middle_outer = 10/2;
  // double ntheta_upper_outer = 8/2;

  double zinner = 1.22;
  double zouter = 1.5;
  double rright_out = 1.6;

  double Lz = (M_PI-1e-14)*2.0;
  double theta_lo = -Lz/2.0, theta_up = Lz/2.0;

  // block 0. Lower outer PF region.
  gkyl_gk_block_geom_set_block(bgeom, 0, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_pf, theta_lo},
      .upper = { psi_up_pf, theta_up},
      .cells = { npsi_pf, ntheta_lower_outer },
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_PF_LO_R,
          .half_domain = true,
          .rright = rright_out,
          .rleft = 0.2,
          .rmin = 0.2,
          .rmax = 1.6,
          .zmin_right = -zouter,
          .zmin_left = -zinner,
          .plate_spec = true,
          .plate_func_lower = shaped_pfunc_lower_outer,
          .plate_func_upper = shaped_pfunc_lower_inner,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL},
        { .bid = 1, .dir = 0, .edge = GKYL_LOWER_POSITIVE}  // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL}, // physical boundary
        { .bid = 5, .dir = 1, .edge = GKYL_LOWER_POSITIVE}
      }
    }
  );

  // block 1. Lower outer SOL.
  gkyl_gk_block_geom_set_block(bgeom, 1, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_outer_sol, theta_lo},
      .upper = { psi_up_outer_sol,  theta_up},
      .cells = { npsi_outer_sol, ntheta_lower_outer},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        //.geometry_id = GKYL_GEOMETRY_FROMFILE,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_DN_SOL_OUT_LO,
          .half_domain = true,
          .rclose = 1.6,       // Closest R to region of interest
          .rright = rright_out,       // Closest R to outboard SOL
          .rleft = 0.2,        // closest R to inboard SOL
          .rmin = 0.2,         // smallest R in machine
          .rmax = 1.6,         // largest R in machine
          .use_cubics = false, // Whether to use cubic representation of psi(R,Z) for field line tracing
          .zmin = -zouter,
          .zmax = zouter,
          .plate_spec = true,
          .plate_func_lower = shaped_pfunc_lower_outer,
          .plate_func_upper = shaped_pfunc_upper_outer,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 0, .dir = 0, .edge = GKYL_UPPER_POSITIVE}, // physical boundary
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}, // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL}, // physical boundary
        { .bid = 2, .dir = 1, .edge = GKYL_LOWER_POSITIVE},
      }
    }
  );

  // block 2. Middle outer SOL.
  gkyl_gk_block_geom_set_block(bgeom, 2, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_outer_sol, theta_lo },
      .upper = { psi_up_outer_sol, theta_up },
      .cells = { npsi_outer_sol, ntheta_middle_outer},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        //.geometry_id = GKYL_GEOMETRY_FROMFILE,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_DN_SOL_OUT_MID,
          .half_domain = true,
          .rclose = 1.6,       // Closest R to region of interest
          .rright = rright_out,       // Closest R to outboard SOL
          .rleft = 0.2,        // closest R to inboard SOL
          .rmin = 0.2,         // smallest R in machine
          .rmax = 1.6,         // largest R in machine
          .use_cubics = false, // Whether to use cubic representation of psi(R,Z) for field line tracing
          .zmin = -zouter,
          .zmax = zouter,
          .plate_spec = true,
          .plate_func_lower = shaped_pfunc_lower_outer,
          .plate_func_upper = shaped_pfunc_upper_outer,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 6, .dir = 0, .edge = GKYL_UPPER_POSITIVE}, // physical boundary
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}, // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 1, .dir = 1, .edge = GKYL_UPPER_POSITIVE},
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL},
      }
    }
  );
  
  // block 7. Middle inner SOL.
  gkyl_gk_block_geom_set_block(bgeom, 3, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_inner_sol, theta_lo },
      .upper = { psi_up_inner_sol, theta_up },
      .cells = { npsi_inner_sol, ntheta_middle_inner},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_DN_SOL_IN_MID,
          .half_domain = true,
          .rleft = 0.2,
          .rright= rright_out,
          .rmin = 0.2,
          .rmax = 1.6,
          .zmin = -zinner,  
          .zmax = zinner,  
          .plate_spec = true,
          .plate_func_upper = shaped_pfunc_upper_inner,
          .plate_func_lower= shaped_pfunc_lower_inner,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 7, .dir = 0, .edge = GKYL_UPPER_POSITIVE}, // physical boundary
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}
      },
      .connections[1] = { // z-direction connections
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL},
        { .bid = 4, .dir = 1, .edge = GKYL_LOWER_POSITIVE}
      }
    }
  );

  // block 8. Lower inner SOL.
  gkyl_gk_block_geom_set_block(bgeom, 4, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_inner_sol, theta_lo },
      .upper = { psi_up_inner_sol, theta_up },
      .cells = { npsi_inner_sol, ntheta_lower_inner},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_DN_SOL_IN_LO,
          .half_domain = true,
          .rleft = 0.2,
          .rright= rright_out,
          .rmin = 0.2,
          .rmax = 1.6,
          .zmin = -zinner,  
          .zmax = zinner,  
          .plate_spec = true,
          .plate_func_upper = shaped_pfunc_upper_inner,
          .plate_func_lower= shaped_pfunc_lower_inner,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 5, .dir = 0, .edge = GKYL_UPPER_POSITIVE}, // physical boundary
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}
      },
      .connections[1] = { // z-direction connections
        { .bid = 3, .dir = 1, .edge = GKYL_UPPER_POSITIVE},
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL}
      }
    }
  );

  // block 9. Lower inner PF region.
  gkyl_gk_block_geom_set_block(bgeom, 5, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_pf, theta_lo},
      .upper = { psi_up_pf, theta_up},
      .cells = { npsi_pf, ntheta_lower_inner },
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_PF_LO_L,
          .half_domain = true,
          .rright = rright_out,
          .rleft = 0.2,
          .rmin = 0.2,
          .rmax = 1.6,
          .zmin_right = -zouter,
          .zmin_left = -zinner,
          .plate_spec = true,
          .plate_func_lower = shaped_pfunc_lower_outer,
          .plate_func_upper = shaped_pfunc_lower_inner,
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL},
        { .bid = 4, .dir = 0, .edge = GKYL_LOWER_POSITIVE}  // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 0, .dir = 1, .edge = GKYL_UPPER_POSITIVE},
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL} // physical boundary
      }
    }
  );


  // block 10. outer core.
  gkyl_gk_block_geom_set_block(bgeom, 6, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_core, theta_lo},
      .upper = { psi_up_core,  theta_up},
      .cells = { npsi_core, ntheta_middle_outer},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_CORE_R,
          .half_domain = true,
          .rclose = 1.6,       // Closest R to region of interest
          .rright = rright_out,       // Closest R to outboard SOL
          .rleft = 0.2,        // closest R to inboard SOL
          .rmin = 0.2,         // smallest R in machine
          .rmax = 1.6,         // largest R in machine
          .use_cubics = false, // Whether to use cubic representation of psi(R,Z) for field line tracing
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}, // physical boundary
        { .bid = 2, .dir = 0, .edge = GKYL_LOWER_POSITIVE}, // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 7, .dir = 1, .edge = GKYL_UPPER_POSITIVE}, // physical boundary
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL},
      }
    }
  );

  // block 11. Inner Core.
  gkyl_gk_block_geom_set_block(bgeom, 7, &(struct gkyl_gk_block_geom_info) {
      .lower = { psi_lo_core, theta_lo },
      .upper = { psi_up_core, theta_up },
      .cells = { npsi_core, ntheta_middle_inner},
      .cuts = { 1, 1 },
      .geometry = {
        .world = {0.0},
        .geometry_id = GKYL_TOKAMAK,
        .efit_info = efit_inp,
        .tok_grid_info = (struct gkyl_tok_geo_grid_inp) {
          .ftype = GKYL_CORE_L,
          .half_domain = true,
          .rclose = 0.2,       // Closest R to region of interest
          .rright = rright_out,       // Closest R to outboard SOL
          .rleft = 0.2,        // closest R to inboard SOL
          .rmin = 0.2,         // smallest R in machine
          .rmax = 1.6,         // largest R in machine
          .use_cubics = false, // Whether to use cubic representation of psi(R,Z) for field line tracing
        }
      },
      
      .connections[0] = { // x-direction connections
        { .bid = 0, .dir = 0, .edge = GKYL_PHYSICAL}, // physical boundary
        { .bid = 3, .dir = 0, .edge = GKYL_LOWER_POSITIVE}, // physical boundary
      },
      .connections[1] = { // z-direction connections
        { .bid = 0, .dir = 1, .edge = GKYL_PHYSICAL},
        { .bid = 6, .dir = 1, .edge = GKYL_LOWER_POSITIVE},
      }
    }
  );

  return bgeom;
  // printf("Created block geom\n");
}

struct gk_step_ctx {
  int cdim, vdim; // Dimensionality.
  double chargeElc; // electron charge
  double massElc; // electron mass
  double chargeIon; // ion charge
  double massIon; // ion mass
  double massH0; // Hydrogen mass
  double Te; // electron temperature
  double Ti; // ion temperature
  double TH0; // neutral hydrogen temperature
  double vtIon;
  double vtElc;
  double vtH0;
  double nuElc; // electron collision frequency
  double nuIon; // ion collision frequency
  double nuFrac; // Factor to multiply collision frequencies
  double B0; // reference magnetic field
  double n0; // reference density
  double n0H0; // neutral hydrogen reference density
  double diffusionD; // Anomalous particle diffusivity.
  // Source parameters
  double Pin; // Input power.
  double nsource;
  double Tsource;
  // Simulation parameters
  int Nx; // Cell count (configuration space: x-direction).
  int Nz; // Cell count (configuration space: z-direction).
  int Nvpar; // Cell count (velocity space: parallel velocity direction).
  int Nmu; // Cell count (velocity space: magnetic moment direction).
  int cells[GKYL_MAX_DIM]; // Number of cells in all directions.
  double vpar_max_elc; // Velocity space extents in vparallel for electrons
  double mu_max_elc; // Velocity space extents in mu for electrons
  double vpar_max_ion; // Velocity space extents in vparallel for ions
  double mu_max_ion; // Velocity space extents in mu for ions
  double vpar_max_H0; // Velocity space extents in vparallel for H0
  double t_end; // end time
  int num_frames; // number of output frames
  double write_phase_freq; // Frequency of writing phase-space diagnostics (as a fraction of num_frames).
  int int_diag_calc_num; // Number of integrated diagnostics computations (=INT_MAX for every step).
  double dt_failure_tol; // Minimum allowable fraction of initial time-step.
  int num_failures_max; // Maximum allowable number of consecutive small time-steps.
};



struct gk_step_ctx
create_ctx(struct gkyl_app_args *app_args)
{
  int cdim = 2, vdim = 2; // Dimensionality.

  double eps0 = GKYL_EPSILON0;
  double eV = GKYL_ELEMENTARY_CHARGE;
  double mi = 2.014*GKYL_PROTON_MASS; // ion mass
  double mH0 = GKYL_PROTON_MASS; // H0 mass
  double me = GKYL_ELECTRON_MASS;
  double qi = eV; // ion charge
  double qe = -eV; // electron charge

  // Reference input power. Obtained with:
  //   half_domain = true
  //   temp_fac = 3
  //   Ti = Te = 300.0*eV/temp_fac
  //   nsource = 1.675e23*2.5*temp_fac
  //   Tsource = 300.0*eV/temp_fac
  // = 3.325975 MW
  double Pin_ref = 2*2*M_PI*0.5*( me*(1.7605626475662941e+35+1.1487112022769204e+35)
                                 +mi*(4.7608360789117243e+31+3.1062943108612647e+31));
  double Pin = Pin_ref; // Input power (provided via command line).
  double temp_fac = 3.0*Pin_ref/Pin;

  double Te = 300.0*eV/temp_fac; // Electron temperature.
  double Ti = 300.0*eV/temp_fac; // Ion temperature.
  double n0 = 1.0e20; //  Reference number density (1 / m^3).

  double TH0 = 100.0*eV; 
  double B0 = 0.65; // Magnetic field magnitude in Tesla
  // double n0 = 1.0e20; // Particle density in 1/m^3
  double n0H0 = n0*1.0e-1; // Particle density in 1/m^3
  
  // Derived parameters.
  double vtIon = sqrt(Ti/mi);
  double vtElc = sqrt(Te/me);
  double vtH0 = sqrt(TH0/mH0);

  double diffusionD = 0.22; // Anomalous particle diffusivity (m^2/s).
  
  // double nsource = 1.675e22*2.5*temp_fac; // Old source density amplitude (10x too low)
  double nsource = 1.675e23*2.5*3.0; // Source density amplitude (1/m^3/s)
  double Tsource = 300.0*eV/temp_fac;

  // Collision parameters.
  double nuFrac = 0.25;
  double logLambdaElc = 6.6 - 0.5*log(n0/1e20) + 1.5*log(Te/eV);
  double nuElc = nuFrac*logLambdaElc*pow(eV, 4.0)*n0/(6.0*sqrt(2.0)*M_PI*sqrt(M_PI)*eps0*eps0*sqrt(me)*(Te*sqrt(Te)));  // collision freq

  double logLambdaIon = 6.6 - 0.5*log(n0/1e20) + 1.5*log(Ti/eV);
  double nuIon = nuFrac*logLambdaIon*pow(eV, 4.0)*n0/(12.0*M_PI*sqrt(M_PI)*eps0*eps0*sqrt(mi)*(Ti*sqrt(Ti)));

  // Simulation box size (m).
  double vpar_max_elc = 8.0*vtElc;
  double mu_max_elc = 18*me*vtElc*vtElc/(2.0*B0);

  double vpar_max_ion = 8.0*vtIon;
  double mu_max_ion = 18*mi*vtIon*vtIon/(2.0*B0);

  double vpar_max_H0 = 6.0*vtH0;

  // Number of cells.
  int Nx = 4;
  int Nz = 8;
  int Nvpar = 32;
  int Nmu = 16;

  double t_end = 1.0e-6; 
  double num_frames = 100;
  double write_phase_freq = 0.2; // Frequency of writing phase-space diagnostics (as a fraction of num_frames).
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-4; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  struct gk_step_ctx ctx = {
    .cdim = cdim,
    .vdim = vdim,
    .Pin = Pin,
    .chargeElc = qe, 
    .massElc = me, 
    .chargeIon = qi, 
    .massIon = mi,
    .massH0 = mH0,
    .Te = Te, 
    .Ti = Ti, 
    .TH0 = TH0, 
    .vtIon = vtIon,
    .vtElc = vtElc,
    .vtH0 = vtH0,
    .nuElc = nuElc, 
    .nuIon = nuIon, 
    .nuFrac = nuFrac,
    .B0 = B0, 
    .n0 = n0, 
    .n0H0 = n0H0,
    .diffusionD = diffusionD,
    .nsource = nsource,
    .Tsource = Tsource,
    .vpar_max_elc = vpar_max_elc, 
    .mu_max_elc = mu_max_elc, 
    .vpar_max_ion = vpar_max_ion, 
    .mu_max_ion = mu_max_ion, 
    .vpar_max_H0 = vpar_max_H0, 
    .Nx = Nx,
    .Nz = Nz,
    .Nvpar = Nvpar,
    .Nmu = Nmu,
    .cells = {Nx, Nz, Nvpar, Nmu},
    .t_end = t_end, 
    .num_frames = num_frames, 
    .write_phase_freq = write_phase_freq,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };
  return ctx;
  // printf("Created context\n");
}

//// NEW UNIFORM DENSITY FUNCTIONS

// void
// init_density(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0; // uniform density everywhere
// }

// void
// init_density_core(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0; // uniform density everywhere
// }

// void
// init_density_outboard_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0H0; // uniform neutral density everywhere
// }

// void
// init_density_inboard_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0H0; // uniform neutral density everywhere
// }

// void
// init_density_pfup_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0H0; // uniform neutral density everywhere
// }

// void
// init_density_pflo_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0H0; // uniform neutral density everywhere
// }

// void
// init_density_empty_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0H0 * 1e-5; // very low uniform neutral density
// }

// void
// init_density_outer(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0; // uniform density everywhere
// }

// void
// init_density_inner(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0; // uniform density everywhere
// }

// void
// init_density_pf(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
// {
//   struct gk_step_ctx *app = ctx;
//   fout[0] = app->n0; // uniform density everywhere
// }

//// OLD SPATIALLY VARYING DENSITY FUNCTIONS

void
init_density(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];

  struct gk_step_ctx *app = ctx;
  double n0 = app->n0;
  fout[0] = n0;
}

void
init_density_core(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
    // Density profile: 2e19 at inner core, 1e19 at separatrix (psi)
  double psi = xn[0]; // psi is the first coordinate

  // Get psi boundaries from block geometry (update these if needed)
  // Example: psi_lo_core = inner core, psi_up_core = separatrix
  double psi_lo_core =  -0.0354402478890806-0.0069402478890806/4.0; // inner core
  double psi_up_core =  -0.0354402478890806;   // separatrix (example value, use your actual psisep)

  double n_inner = 2.0e20;
  double n_sep   = 1.0e20;

  // Calculate slope and intercept for linear profile
  double slope = (n_inner - n_sep) / (psi_lo_core - psi_up_core);
  double intercept = n_inner - slope * psi_lo_core;

  double n = slope * psi + intercept;
  fout[0] = n;
}

void
init_density_outboard_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];
  struct gk_step_ctx *app = ctx;
  double cz = 5.640389838180728;
  double n = app->n0H0;
  if (z < 0.0)
    n = n*exp(-cz*(z+M_PI));
  else 
    n = n*exp(-cz*(M_PI-z));
  fout[0] = n;
}

void
init_density_inboard_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];
  struct gk_step_ctx *app = ctx;
  double cz = 10.166078433144026;
  double n = app->n0H0;
  if (z < 0.0)
    n = n*exp(-cz*(z+M_PI));
  else 
    n = n*exp(-cz*(M_PI-z));
  fout[0] = n;
}

void
init_density_pfup_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];
  struct gk_step_ctx *app = ctx;
  double n = app->n0H0;
  if (z < -1.467030) {
    double cz = 4.125110078248697;
    n = n*exp(-cz*(z+M_PI));
  }
  else {
    double cz = 1.4988763017083815;
    n = n*exp(-cz*(M_PI - z));
  }
  fout[0] = n;
}

void
init_density_pflo_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];
  struct gk_step_ctx *app = ctx;
  double n = app->n0H0;
  if (z < 1.467030) {
    double cz = 1.4988763017083815;
    n = n*exp(-cz*(z+M_PI));
  }
  else {
    double cz = 4.125110078248697;
    n = n*exp(-cz*(M_PI - z));
  }
  fout[0] = n;
}


void
init_density_empty_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *app = ctx;
  double n = app->n0H0*1e-5;
  fout[0] = n;
}


void
init_density_outer(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  // Density profile: 1e19 at separatrix, 1e17 at outer boundary
  double psi = xn[0];

  double psisep = -0.0354402478890806;
  double psi_outer = psisep + 0.0069402478890806/4.0;

  double n_sep = 1.0e20;
  double n_outer = 1.0e18;

  double slope = (n_sep - n_outer) / (psisep - psi_outer);
  double intercept = n_sep - slope * psisep;

  double n = slope * psi + intercept;
  fout[0] = n;
}

void
init_density_inner(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  // Density profile: 1e19 at separatrix, 1e17 at outer boundary
  double psi = xn[0];

  double psisep = -0.0354402478890806;
  double psi_outer = psisep + 0.0069402478890806/4.0;

  double n_sep = 1.0e20;
  double n_outer = 1.0e18;

  double slope = (n_sep - n_outer) / (psisep - psi_outer);
  double intercept = n_sep - slope * psisep;

  double n = slope * psi + intercept;
  fout[0] = n;
}

void
init_density_pf(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  // Density profile: 1e19 at separatrix, 1e17 at outer boundary
  double psi = xn[0];

  double psisep = -0.0354402478890806;
  double psi_outer = psisep + 0.0069402478890806/4.0;

  double n_sep = 1.0e20;
  double n_outer = 1.0e18;

  double slope = (n_sep - n_outer) / (psisep - psi_outer);
  double intercept = n_sep - slope * psisep;

  double n = slope * psi + intercept;
  fout[0] = n;

}


void
source_density(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[1];

  struct gk_step_ctx *app = ctx;
  double nsource = app->nsource;
  if(x <= -0.0369584)
    fout[0] = nsource;
  else 
    fout[0] = nsource*1.0e-5;

  // printf("Initialized densities\n");
}


void
init_upar(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

void
init_udrift_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
  fout[1] = 0.0;
  fout[2] = 0.0;
}

void
init_temp_elc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *app = ctx;
  double T = 2.0*app->Te;
  fout[0] = T;
}

void
init_temp_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *app = ctx;
  double T = 2.0*app->Ti;
  fout[0] = T;
}
void
source_temp(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *app = ctx;
  double T = app->Tsource;
  fout[0] = T;
}


void
init_temp_H0(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *app = ctx;
  double T = app->TH0;
  fout[0] = T;
}

void
init_nu_elc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *input = ctx;
  fout[0] = input->nuElc;
}

void
init_nu_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *input = ctx;
  fout[0] = input->nuIon;
}

void
evalNuElc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *input = ctx;
  fout[0] = input->nuElc;
}

void
evalNuIon(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_step_ctx *input = ctx;
  fout[0] = input->nuIon;
}

static inline void
mapc2p_vel_elc(double t, const double* GKYL_RESTRICT vc, double* GKYL_RESTRICT vp, void* ctx)
{
  struct gk_step_ctx *app = ctx;
  double cvpar = vc[0], cmu = vc[1];

  double mu_max_elc = app->mu_max_elc;
  double vpar_max_elc = app->vpar_max_elc;

  double mu = 0.0;
  double vpar = 0.0;

  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vpar = vpar_max_elc*cvpar;
  else if (cvpar < -0.5)
    vpar = -vpar_max_elc*2.0*pow(cvpar,2);
  else
    vpar =  vpar_max_elc*2.0*pow(cvpar,2);

  mu = mu_max_elc * (cmu * cmu);

  // Set rescaled electron velocity space coordinates (vpar, mu) from old velocity space coordinates (cvpar, cmu):
  vp[0] = vpar; vp[1] = mu;
}

static inline void
mapc2p_vel_ion(double t, const double* GKYL_RESTRICT vc, double* GKYL_RESTRICT vp, void* ctx)
{
  struct gk_step_ctx *app = ctx;
  double cvpar = vc[0], cmu = vc[1];

  double mu_max_ion = app->mu_max_ion;
  double vpar_max_ion = app->vpar_max_ion;

  double mu = 0.0;
  double vpar = 0.0;

  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vpar = vpar_max_ion*cvpar;
  else if (cvpar < -0.5)
    vpar = -vpar_max_ion*2.0*pow(cvpar,2);
  else
    vpar =  vpar_max_ion*2.0*pow(cvpar,2);

  mu = mu_max_ion * (cmu * cmu);

  // Set rescaled ion velocity space coordinates (vpar, mu) from old velocity space coordinates (cvpar, cmu):
  vp[0] = vpar ; vp[1] = mu;
}

void
diffusion_D_func(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void* ctx)
{
  struct gk_step_ctx *app = ctx;

  fout[0] = app->diffusionD; // Diffusivity [m^2/s].
}

void
calc_integrated_diagnostics(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_multib_app* app,
  double t_curr, bool is_restart_IC, bool force_calc, double dt)
{
  if (!is_restart_IC && (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc)) {
    gkyl_gyrokinetic_multib_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_multib_app_calc_integrated_mom(app, t_curr);

    if ( !(dt < 0.0) )
      gkyl_gyrokinetic_multib_app_save_dt(app, t_curr, dt);
  }
}

void
write_data(struct gkyl_tm_trigger* iot_conf, struct gkyl_tm_trigger* iot_phase,
  gkyl_gyrokinetic_multib_app* app, double t_curr, bool is_restart_IC, bool force_write)
{
  bool trig_now_conf = gkyl_tm_trigger_check_and_bump(iot_conf, t_curr);
  if (trig_now_conf || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;
    gkyl_gyrokinetic_multib_app_write_conf(app, t_curr, frame);

    if (!is_restart_IC) {
      gkyl_gyrokinetic_multib_app_write_field_energy(app);
      gkyl_gyrokinetic_multib_app_write_integrated_mom(app);
      gkyl_gyrokinetic_multib_app_write_dt(app);
    }
  }

  bool trig_now_phase = gkyl_tm_trigger_check_and_bump(iot_phase, t_curr);
  if (trig_now_phase || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;

    gkyl_gyrokinetic_multib_app_write_phase(app, t_curr, frame);
  }
}

int
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Init(&argc, &argv);
  }
#endif

  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct gk_step_ctx ctx = create_ctx(&app_args); // Context for init functions.

  // Construct block geometry.
  struct gkyl_gk_block_geom *bgeom = create_gk_block_geom();
  // printf("Created block geom\n");
  int nblocks = gkyl_gk_block_geom_num_blocks(bgeom);

  int cells_x[ctx.cdim], cells_v[ctx.vdim];
  for (int d=0; d<ctx.cdim; d++)
    cells_x[d] = APP_ARGS_CHOOSE(app_args.xcells[d], ctx.cells[d]);
  for (int d=0; d<ctx.vdim; d++)
    cells_v[d] = APP_ARGS_CHOOSE(app_args.vcells[d], ctx.cells[ctx.cdim+d]);

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_gyrokinetic_comms_new(app_args.use_mpi, app_args.use_gpu, stderr);
  
  // Elc Species
  // all data is common across blocks
  struct gkyl_gyrokinetic_multib_species_pb elc_blocks[8];
  elc_blocks[0] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 0,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };

  elc_blocks[1] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 1,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };

  elc_blocks[2] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 2,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };
  elc_blocks[3] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 3,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };
  elc_blocks[4] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 4,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };
  elc_blocks[5] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 5,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

  };

  elc_blocks[6] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 6,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_core,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .ctx_density = &ctx,
        .density = source_density,
        .ctx_upar = &ctx,
        .upar= init_upar,
        .ctx_temp = &ctx,
        .temp = source_temp, 
      }, 
      .diagnostics = {
        .num_diag_moments = 2,
        .diag_moments = { GKYL_F_MOMENT_M0M1M2, GKYL_F_MOMENT_BIMAXWELLIAN, },
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = { GKYL_F_MOMENT_M0M1M2 },
      }
    },

  };

  elc_blocks[7] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 7,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_core,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_elc,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .ctx_density = &ctx,
        .density = source_density,
        .ctx_upar = &ctx,
        .upar= init_upar,
        .ctx_temp = &ctx,
        .temp = source_temp, 
      }, 
      .diagnostics = {
        .num_diag_moments = 2,
        .diag_moments = { GKYL_F_MOMENT_M0M1M2, GKYL_F_MOMENT_BIMAXWELLIAN, },
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = { GKYL_F_MOMENT_M0M1M2 },
      }
    },

  };

  struct gkyl_gyrokinetic_bc elc_phys_bcs[] = {
     // block 0 BCs
    { .bidx = 0, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 0, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 1 BCs
    { .bidx = 1, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 1, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 2 BCs
    { .bidx = 2, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 2, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
    // block 3 BCs
    { .bidx = 3, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 3, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},

    // block 4 BCs
    { .bidx = 4, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 4, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 5 BCs
    { .bidx = 5, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 5, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH },

    //block 10 BCs
    { .bidx = 6, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ZERO_FLUX},
    { .bidx = 6, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
    //block 11 BCs
    { .bidx = 7, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ZERO_FLUX},
    { .bidx = 7, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
  };


  struct gkyl_gyrokinetic_multib_species elc = {
    .name = "elc",
    .charge = ctx.chargeElc, .mass = ctx.massElc,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0}, 
    .cells = { cells_v[0], cells_v[1] },

    .mapc2p = {
      .mapping = mapc2p_vel_elc,
      .ctx = &ctx,
    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES_NO_BY,
    },

    // .collisions =  {
    //   .collision_id = GKYL_LBO_COLLISIONS,
    //   .nu_frac = ctx.nuFrac,
    //   .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
    //   .temp_ref = ctx.Te, // Temperature used to calculate coulomb logarithm
    //   .num_cross_collisions = 1,
    //   .collide_with = { "ion" },
    // },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Te, // Temperature used to calculate coulomb logarithm
      .is_implicit = true,
      .num_cross_collisions = 1,
      .collide_with = { "ion" },
      .write_diagnostics = true, 
    },

    .anomalous_diffusion = {
      .anomalous_diff_id = GKYL_GK_ANOMALOUS_DIFF_D,
      .D_profile = diffusion_D_func,
      .D_profile_ctx = &ctx,
    }, 

    .num_diag_moments = 7,
    .diag_moments = { GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP },
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .time_rate_diagnostics = true,
    .boundary_flux_diagnostics = {
      .num_diag_moments = 1,
      .diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
//      .time_integrated = true,
    },


    .duplicate_across_blocks = false,
    .blocks = elc_blocks,
    .num_physical_bcs = 16,
    .bcs = elc_phys_bcs,
  };


  // Ion Species
  struct gkyl_gyrokinetic_multib_species_pb ion_blocks[8];
  ion_blocks[0] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 0,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };

  ion_blocks[1] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 1,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };

  ion_blocks[2] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 2,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };

  ion_blocks[3] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 3,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_outer,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };
  ion_blocks[4] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 4,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };
  ion_blocks[5] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 5,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_pf,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

  };

  ion_blocks[6] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 6,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_core,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .ctx_density = &ctx,
        .density = source_density,
        .ctx_upar = &ctx,
        .upar= init_upar,
        .ctx_temp = &ctx,
        .temp = source_temp, 
      }, 
      .diagnostics = {
        .num_diag_moments = 2,
        .diag_moments = { GKYL_F_MOMENT_M0M1M2, GKYL_F_MOMENT_BIMAXWELLIAN, },
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = { GKYL_F_MOMENT_M0M1M2 },
      }
    },

  };

 ion_blocks[7] = (struct gkyl_gyrokinetic_multib_species_pb) {

    .block_id = 7,

    .polarization_density = ctx.n0,

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .density = init_density_core,
      .ctx_upar = &ctx,
      .upar = init_upar,
      .ctx_temp = &ctx,
      .temp = init_temp_ion,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = 1,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM, 
        .ctx_density = &ctx,
        .density = source_density,
        .ctx_upar = &ctx,
        .upar= init_upar,
        .ctx_temp = &ctx,
        .temp = source_temp, 
      }, 
      .diagnostics = {
        .num_diag_moments = 2,
        .diag_moments = { GKYL_F_MOMENT_M0M1M2, GKYL_F_MOMENT_BIMAXWELLIAN, },
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = { GKYL_F_MOMENT_M0M1M2 },
      }
    },

  };


  struct gkyl_gyrokinetic_bc ion_phys_bcs[] = {
    // block 0 BCs
    { .bidx = 0, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 0, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 1 BCs
    { .bidx = 1, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 1, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 2 BCs
    { .bidx = 2, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 2, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
    // block 3 BCs
    { .bidx = 3, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB },
    { .bidx = 3, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},

    // block 4 BCs
    { .bidx = 4, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 4, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH},
    // block 5 BCs
    { .bidx = 5, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
    { .bidx = 5, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH },

    //block 10 BCs
    { .bidx = 6, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ZERO_FLUX},
    { .bidx = 6, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
    //block 11 BCs
    { .bidx = 7, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ZERO_FLUX},
    { .bidx = 7, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_REFLECT},
  };

  struct gkyl_gyrokinetic_multib_species ion = {
    .name = "ion",
    .charge = ctx.chargeIon, .mass = ctx.massIon,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0}, 
    .cells = { cells_v[0], cells_v[1] },

    .mapc2p = {
      .mapping = mapc2p_vel_ion,
      .ctx = &ctx,
    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES_NO_BY,
    },

    // .collisions =  {
    //   .collision_id = GKYL_LBO_COLLISIONS,
    //   .nu_frac = ctx.nuFrac,
    //   .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
    //   .temp_ref = ctx.Ti, // Temperature used to calculate coulomb logarithm
    //   .num_cross_collisions = 1,
    //   .collide_with = { "elc" },
    // },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Ti, // Temperature used to calculate coulomb logarithm
      .is_implicit = true,
      .num_cross_collisions = 1,
      .collide_with = { "elc" },
      .write_diagnostics = true, 
    },

    .anomalous_diffusion = {
      .anomalous_diff_id = GKYL_GK_ANOMALOUS_DIFF_D,
      .D_profile = diffusion_D_func,
      .D_profile_ctx = &ctx,
    }, 
  
    .num_diag_moments = 7,
    .diag_moments = { GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP },
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .time_rate_diagnostics = true,
    .boundary_flux_diagnostics = {
      .num_diag_moments = 1,
      .diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  
    .duplicate_across_blocks = false,
    .blocks = ion_blocks,
    .num_physical_bcs = 16,
    .bcs = ion_phys_bcs,
  };

//// ALL RELATED TO NEUTRALS IS COMMENTED OUT FOR NOW


  // struct gkyl_gyrokinetic_block_physical_bcs H0_phys_bcs[] = {
  //   // block 0 BCs
  //   { .bidx = 0, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 0, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 0, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 0, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 1, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 1, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 1, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 1, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 2, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 2, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 2, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 2, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 3, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 3, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 3, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 3, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 4, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 4, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 4, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 4, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 5, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 5, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 5, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 5, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 6, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 6, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 6, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 6, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   { .bidx = 7, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 7, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 7, .dir = 1, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},
  //   { .bidx = 7, .dir = 1, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB},

  //   };

  // struct gkyl_gyrokinetic_multib_neut_species_pb H0_blocks[8];
  // H0_blocks[0] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 0,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_pflo_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };

  // H0_blocks[1] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 1,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_outboard_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };

  // H0_blocks[2] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 2,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_empty_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };

  // H0_blocks[3] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 3,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_outboard_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };
  // H0_blocks[4] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 4,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_pfup_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };
  // H0_blocks[5] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 5,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_pfup_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };
  // H0_blocks[6] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 6,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_inboard_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // };
  // H0_blocks[7] = (struct gkyl_gyrokinetic_multib_neut_species_pb) {

  //   .block_id = 7,


  //   .projection = {
  //     .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
  //     .ctx_density = &ctx,
  //     .density = init_density_empty_H0,
  //     .ctx_udrift = &ctx,
  //     .udrift = init_udrift_H0,
  //     .ctx_temp = &ctx,
  //     .temp = init_temp_H0,
  //   },

  // }; 



  // struct gkyl_gyrokinetic_multib_neut_species H0 = {
  //   .name = "H0",
  //   .mass = ctx.massH0,
  //   .lower = { -ctx.vpar_max_H0, -ctx.vpar_max_H0, -ctx.vpar_max_H0},
  //   .upper = {  ctx.vpar_max_H0, ctx.vpar_max_H0, ctx.vpar_max_H0}, 
  //   .cells = { 8, 8, 8 },
  //   .is_static = true,
  //   .num_diag_moments = 3,
  //   .diag_moments = { GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2},

  //   //.react_neut = {
  //   //  .num_react = 3,
  //   //  .react_type = {
  //   //    { .react_id = GKYL_REACT_CX,
  //   //      .type_self = GKYL_SELF_PARTNER,
  //   //      .ion_id = GKYL_ION_H,
  //   //      .elc_nm = "elc",
  //   //      .ion_nm = "ion",
  //   //      .partner_nm = "H0",
  //   //      .ion_mass = ctx.massIon,
  //   //      .partner_mass = ctx.massIon,
  //   //    },
  //   //    { .react_id = GKYL_REACT_IZ,
  //   //      .type_self = GKYL_SELF_DONOR,
  //   //      .ion_id = GKYL_ION_H,
  //   //      .elc_nm = "elc",
  //   //      .ion_nm = "ion", // ion is always the higher charge state
  //   //      .donor_nm = "H0", // interacts with elc to give up charge
  //   //      .charge_state = 0, // corresponds to lower charge state (donor)
  //   //      .ion_mass = ctx.massIon,
  //   //      .elc_mass = ctx.massElc,
  //   //    },
  //   //    { .react_id = GKYL_REACT_RECOMB,
  //   //      .type_self = GKYL_SELF_RECVR,
  //   //      .ion_id = GKYL_ION_H,
  //   //      .elc_nm = "elc",
  //   //      .ion_nm = "ion",
  //   //      .recvr_nm = "H0",
  //   //      .charge_state = 0,
  //   //      .ion_mass = ctx.massIon,
  //   //      .elc_mass = ctx.massElc,
  //   //    },
  //   //  },
  //   //},

  //   .duplicate_across_blocks = false,
  //   .blocks = H0_blocks,
  //   .num_physical_bcs = 32,
  //   .bcs = H0_phys_bcs,
  // };

  // printf("Done setting up species\n");



  // Field object
  struct gkyl_gyrokinetic_multib_field_pb field_blocks[1];
  field_blocks[0] = (struct gkyl_gyrokinetic_multib_field_pb) {
    // .polarization_bmag = 0.65,
    .time_rate_diagnostics = true,
  };

  struct gkyl_gyrokinetic_bc field_phys_bcs[] = {
    // block 1 BCs
    { .bidx = 1, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },
    // block 2 BCs
    { .bidx = 2, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },

    // block 3 BCs
    { .bidx = 3, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },
    // block 4 BCs
    { .bidx = 4, .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },

    // block 0 BCs
    { .bidx = 0, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_NEUMANN, .value = {0.0} },
    // block 5 BCs
    { .bidx = 5, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_NEUMANN, .value = {0.0} },

    // block 6 BCs
    { .bidx = 6, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_NEUMANN, .value = {0.0} },
    // block 7 BCs
    { .bidx = 7, .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_NEUMANN, .value = {0.0} },
  };

  struct gkyl_gyrokinetic_multib_field field = {
    .duplicate_across_blocks = true,
    .blocks = field_blocks, 
    .num_physical_bcs = 8,
    // .half_domain = true, 
    .bcs = field_phys_bcs,
    .time_rate_diagnostics = true,
  };

  // printf("Created field\n");

  struct gkyl_gyrokinetic_multib app_inp = {
    .name = "gk_multib_nstxu_2x2v_p1",

    .cdim = ctx.cdim,
    .poly_order = 1,
    .basis_type = app_args.basis_type,
    .use_gpu = app_args.use_gpu,
    .cfl_frac = 1.0,
    // .cfl_frac_omegaH = 1.7,

    .gk_block_geom = bgeom,
    
    .num_species = 2,
    .species = { elc, ion},

    .num_neut_species = 0,
    .neut_species = {  },

    .field = field,
    //.skip_field=true,

    .comm = comm
  };

  // Create app object.
  struct gkyl_gyrokinetic_multib_app *app = gkyl_gyrokinetic_multib_app_new(&app_inp);

  // Initial and final simulation times.
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.t_end;
  // Initialize simulation.
  if (app_args.is_restart) {
    struct gkyl_app_restart_status status = gkyl_gyrokinetic_multib_app_read_from_frame(app, app_args.restart_frame);

    if (status.io_status != GKYL_ARRAY_RIO_SUCCESS) {
      gkyl_gyrokinetic_multib_app_cout(app, stderr, "*** Failed to read restart file! (%s)\n",
        gkyl_array_rio_status_msg(status.io_status));
      goto freeresources;
    }

    frame_curr = status.frame;
    t_curr = status.stime;

    gkyl_gyrokinetic_multib_app_cout(app, stdout, "Restarting from frame %d", frame_curr);
    gkyl_gyrokinetic_multib_app_cout(app, stdout, " at time = %g\n", t_curr);
  }
  else {
    gkyl_gyrokinetic_multib_app_apply_ic(app, t_curr);
  }

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write_conf = { .dt = t_end/num_frames, .tcurr = t_curr, .curr = frame_curr };
  struct gkyl_tm_trigger trig_write_phase = { .dt = t_end/(ctx.write_phase_freq*num_frames), .tcurr = t_curr, .curr = frame_curr};
  struct gkyl_tm_trigger trig_calc_intdiag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc),
    .tcurr = t_curr, .curr = frame_curr };

  // Write out ICs (if restart, it overwrites the restart frame).
  calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, app_args.is_restart, false, -1.0);
  write_data(&trig_write_conf, &trig_write_phase, app, t_curr, app_args.is_restart, false);

  printf("Starting simulation ...\n");

  double dt = t_end-t_curr; // Initial time step.
  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  long step = 1;
  while ((t_curr < t_end) && (step <= app_args.num_steps)) {
    gkyl_gyrokinetic_multib_app_cout(app, stdout, "Taking time-step %ld at t = %g ...", step, t_curr);
    struct gkyl_update_status status = gkyl_gyrokinetic_multib_update(app, dt);
    gkyl_gyrokinetic_multib_app_cout(app, stdout, " dt = %g\n", status.dt_actual);

    if (!status.success) {
      gkyl_gyrokinetic_multib_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }

    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false, t_curr > t_end, status.dt_actual);
    write_data(&trig_write_conf, &trig_write_phase, app, t_curr, false, t_curr > t_end);

    if (dt_init < 0.0) {
      dt_init = status.dt_actual;
    }
    else if (status.dt_actual < dt_failure_tol * dt_init) {
      num_failures += 1;

      gkyl_gyrokinetic_multib_app_cout(app, stdout, "WARNING: Time-step dt = %g", status.dt_actual);
      gkyl_gyrokinetic_multib_app_cout(app, stdout, " is below %g*dt_init ...", dt_failure_tol);
      gkyl_gyrokinetic_multib_app_cout(app, stdout, " num_failures = %d\n", num_failures);
      if (num_failures >= num_failures_max) {
        gkyl_gyrokinetic_multib_app_cout(app, stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
        gkyl_gyrokinetic_multib_app_cout(app, stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
        calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false, true, status.dt_actual);
        write_data(&trig_write_conf, &trig_write_phase, app, t_curr, false, true);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;
  }

  gkyl_gyrokinetic_multib_app_stat_write(app);

  // Fetch simulation statistics.
  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_multib_app_stat(app);

  gkyl_gyrokinetic_multib_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_multib_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_multib_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_multib_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0) {
    gkyl_gyrokinetic_multib_app_cout(app, stdout, "  Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_multib_app_cout(app, stdout, "  Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }
  gkyl_gyrokinetic_multib_app_cout(app, stdout, "Number of RK stage-3 failures %ld.\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_multib_app_cout(app, stdout, "Number of write calls %ld.\n", stat.n_io);
  gkyl_gyrokinetic_multib_app_print_timings(app, stdout);

freeresources:
  // Free resources after simulation completion.
  gkyl_gyrokinetic_multib_app_release(app);
  gkyl_gyrokinetic_comms_release(comm);
  gkyl_gk_block_geom_release(bgeom);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Finalize();
#endif

  return 0;
}

