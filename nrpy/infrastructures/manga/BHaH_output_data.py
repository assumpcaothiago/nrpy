# nrpy/infrastructures/manga/BHaH_output_data.py
"""
Construct a C function that forces bhah_lib checkpoint and diagnostics output
for a ChaNGa snapshot prefix.

Author: Thiago Assumpcao
        assumpcaothiago **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_BHaH_output_data() -> Union[None, pcg.NRPyEnv_type]:
    """
    Register a helper that forces checkpoint and diagnostics output for a ChaNGa prefix.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
        "diagnostics/diagnostics_nearest_common.h",
        "dirent.h",
        "unistd.h",
    ]

    prefunc = r"""
static int bhah_build_sidecar_filename(char *dst, const size_t dst_sz, const char *prefix_path, const char *label, const int include_grid,
                                       const int grid, const char *recipe, const char *extension) {
  char grid_suffix[32] = "";
  char recipe_suffix[256] = "";

  if (include_grid) {
    const int grid_written = snprintf(grid_suffix, sizeof(grid_suffix), ".grid%02d", grid);
    if (grid_written < 0 || (size_t)grid_written >= sizeof(grid_suffix)) {
      fprintf(stderr, "BHaH_output_data: grid suffix is too long for grid %d\n", grid);
      return 0;
    }
  }

  if (recipe != NULL && recipe[0] != '\0') {
    const int recipe_written = snprintf(recipe_suffix, sizeof(recipe_suffix), ".%s", recipe);
    if (recipe_written < 0 || (size_t)recipe_written >= sizeof(recipe_suffix)) {
      fprintf(stderr, "BHaH_output_data: recipe suffix is too long for recipe '%s'\n", recipe);
      return 0;
    }
  }

  const int written = snprintf(dst, dst_sz, "%s.%s%s%s.%s", prefix_path, label, grid_suffix, recipe_suffix, extension);
  if (written < 0 || (size_t)written >= dst_sz) {
    fprintf(stderr, "BHaH_output_data: destination filename is too long for prefix '%s' and label '%s'\n", prefix_path, label);
    return 0;
  }
  return 1;
}

static int bhah_rename_file(const char *src, const char *dst) {
  if (rename(src, dst) != 0) {
    fprintf(stderr, "BHaH_output_data: could not rename '%s' to '%s': %s\n", src, dst, strerror(errno));
    return 0;
  }
  return 1;
}

static int bhah_try_rename_grid_output(const char *src, const char *prefix_path, const char *label, const int include_grid, const int grid) {
  if (access(src, F_OK) != 0)
    return 0;

  char dst[1024];
  if (!bhah_build_sidecar_filename(dst, sizeof(dst), prefix_path, label, include_grid, grid, NULL, "txt"))
    return -1;

  return bhah_rename_file(src, dst) ? 1 : -1;
}

static int bhah_rename_matching_integral_outputs(const char *match_prefix, const char *prefix_path, const char *label) {
  DIR *dir = opendir(".");
  if (dir == NULL) {
    fprintf(stderr, "BHaH_output_data: could not open current directory while renaming diagnostics: %s\n", strerror(errno));
    return -1;
  }

  const size_t match_prefix_len = strlen(match_prefix);
  int renamed = 0;

  struct dirent *entry = NULL;
  while ((entry = readdir(dir)) != NULL) {
    const char *name = entry->d_name;
    if (strncmp(name, match_prefix, match_prefix_len) != 0)
      continue;

    const char *recipe_with_ext = name + match_prefix_len;
    const size_t recipe_with_ext_len = strlen(recipe_with_ext);
    if (recipe_with_ext_len <= 4 || strcmp(recipe_with_ext + recipe_with_ext_len - 4, ".txt") != 0) {
      fprintf(stderr, "BHaH_output_data: integral output '%s' does not end in .txt\n", name);
      closedir(dir);
      return -1;
    }

    char recipe[256];
    const size_t recipe_len = recipe_with_ext_len - 4;
    if (recipe_len == 0 || recipe_len >= sizeof(recipe)) {
      fprintf(stderr, "BHaH_output_data: invalid recipe name extracted from '%s'\n", name);
      closedir(dir);
      return -1;
    }
    memcpy(recipe, recipe_with_ext, recipe_len);
    recipe[recipe_len] = '\0';

    char dst[1024];
    if (!bhah_build_sidecar_filename(dst, sizeof(dst), prefix_path, label, 0, 0, recipe, "txt")) {
      closedir(dir);
      return -1;
    }

    if (!bhah_rename_file(name, dst)) {
      closedir(dir);
      return -1;
    }
    renamed++;
  }

  closedir(dir);
  return renamed;
}
"""

    desc = (
        "Force bhah_lib checkpoint and diagnostics output for a ChaNGa snapshot "
        "prefix and rename the generated files onto ChaNGa-style sidecar names."
    )
    cfunc_type = "void"
    name = "BHaH_output_data"
    params = "BHaH_struct *bhahstruct, const char *prefix, const REAL output_time"

    body = r"""
if (bhahstruct == NULL || bhahstruct->commondata == NULL || bhahstruct->griddata == NULL) {
  fprintf(stderr, "BHaH_output_data: bhahstruct/commondata/griddata is NULL.\n");
  exit(EXIT_FAILURE);
}
if (prefix == NULL || prefix[0] == '\0') {
  fprintf(stderr, "BHaH_output_data: prefix must be a non-empty string.\n");
  exit(EXIT_FAILURE);
}

commondata_struct *commondata = bhahstruct->commondata;
griddata_struct *griddata = bhahstruct->griddata;

const REAL saved_time = commondata->time;
const REAL saved_checkpoint_every = commondata->checkpoint_every;
const REAL saved_diagnostics_output_every = commondata->diagnostics_output_every;
const int saved_nn = commondata->nn;
const int saved_nn_0 = commondata->nn_0;
const int saved_output_progress_every = commondata->output_progress_every;

const REAL fallback_interval = (fabs(commondata->dt) > 0.0) ? fabs(commondata->dt) : 1.0;
const REAL forced_time = (output_time >= 0.0) ? output_time : saved_time;
const REAL forced_interval = (fabs(forced_time) > 0.0) ? fabs(forced_time) : fallback_interval;

int status = EXIT_SUCCESS;
int renamed_diagnostics = 0;
int existing_out0d = 0;
int existing_out1d_y = 0;
int existing_out1d_z = 0;
int existing_out2d_xy = 0;
int existing_out2d_yz = 0;

commondata->time = forced_time;
commondata->checkpoint_every = forced_interval;
commondata->diagnostics_output_every = forced_interval;
commondata->nn = 0;
commondata->nn_0 = 0;
commondata->output_progress_every = -1;

write_checkpoint(commondata, griddata);
{
  char checkpoint_src[256];
  char checkpoint_dst[1024];
  snprintf(checkpoint_src, sizeof(checkpoint_src), "checkpoint-conv_factor%.2f.dat", commondata->convergence_factor);
  if (!bhah_build_sidecar_filename(checkpoint_dst, sizeof(checkpoint_dst), prefix, "bhah_checkpoint", 0, 0, NULL, "dat") ||
      !bhah_rename_file(checkpoint_src, checkpoint_dst)) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
}

diagnostics(commondata, griddata);

for (int grid = 0; grid < commondata->NUMGRIDS; ++grid) {
  char src[256];
  char coordsys_with_grid[128];

  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "grid%02d-%s", grid, griddata[grid].params.CoordSystemName);
  build_outfile_name(src, sizeof(src), "out0d", coordsys_with_grid, commondata, 0);
  if (access(src, F_OK) == 0)
    existing_out0d++;

  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "%s-grid%02d", griddata[grid].params.CoordSystemName, grid);
  build_outfile_name(src, sizeof(src), "out1d-y", coordsys_with_grid, commondata, 1);
  if (access(src, F_OK) == 0)
    existing_out1d_y++;

  build_outfile_name(src, sizeof(src), "out1d-z", coordsys_with_grid, commondata, 1);
  if (access(src, F_OK) == 0)
    existing_out1d_z++;

  build_outfile_name(src, sizeof(src), "out2d-xy", coordsys_with_grid, commondata, 1);
  if (access(src, F_OK) == 0)
    existing_out2d_xy++;

  build_outfile_name(src, sizeof(src), "out2d-yz", coordsys_with_grid, commondata, 1);
  if (access(src, F_OK) == 0)
    existing_out2d_yz++;
}

for (int grid = 0; grid < commondata->NUMGRIDS; ++grid) {
  char src[256];
  char coordsys_with_grid[128];
  int rc = 0;

  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "grid%02d-%s", grid, griddata[grid].params.CoordSystemName);
  build_outfile_name(src, sizeof(src), "out0d", coordsys_with_grid, commondata, 0);
  rc = bhah_try_rename_grid_output(src, prefix, "bhah_out0d", existing_out0d > 1, grid);
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;

  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "%s-grid%02d", griddata[grid].params.CoordSystemName, grid);
  build_outfile_name(src, sizeof(src), "out1d-y", coordsys_with_grid, commondata, 1);
  rc = bhah_try_rename_grid_output(src, prefix, "bhah_out1d-y", existing_out1d_y > 1, grid);
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;

  build_outfile_name(src, sizeof(src), "out1d-z", coordsys_with_grid, commondata, 1);
  rc = bhah_try_rename_grid_output(src, prefix, "bhah_out1d-z", existing_out1d_z > 1, grid);
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;

  build_outfile_name(src, sizeof(src), "out2d-xy", coordsys_with_grid, commondata, 1);
  rc = bhah_try_rename_grid_output(src, prefix, "bhah_out2d-xy", existing_out2d_xy > 1, grid);
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;

  build_outfile_name(src, sizeof(src), "out2d-yz", coordsys_with_grid, commondata, 1);
  rc = bhah_try_rename_grid_output(src, prefix, "bhah_out2d-yz", existing_out2d_yz > 1, grid);
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;
}

{
  char match_prefix[256];
  snprintf(match_prefix, sizeof(match_prefix), "out3d-integrals-conv_factor%.2f-", commondata->convergence_factor);
  const int rc = bhah_rename_matching_integral_outputs(match_prefix, prefix, "bhah_out3d-integrals");
  if (rc < 0) {
    status = EXIT_FAILURE;
    goto cleanup;
  }
  renamed_diagnostics += rc;
}

if (renamed_diagnostics == 0) {
  fprintf(stderr, "BHaH_output_data: warning: diagnostics() did not emit any files for prefix '%s'.\n", prefix);
}

cleanup:
commondata->time = saved_time;
commondata->checkpoint_every = saved_checkpoint_every;
commondata->diagnostics_output_every = saved_diagnostics_output_every;
commondata->nn = saved_nn;
commondata->nn_0 = saved_nn_0;
commondata->output_progress_every = saved_output_progress_every;

if (status != EXIT_SUCCESS)
  exit(status);
"""

    cfc.register_CFunction(
        include_CodeParameters_h=False,
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
        enable_simd=False,
    )
    return pcg.NRPyEnv()
