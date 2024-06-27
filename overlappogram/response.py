from __future__ import annotations

import numpy as np
from ndcube import NDCube

__all__ = ["prepare_response_function"]


def prepare_response_function(
    response_cube: NDCube, field_angle_range=None, response_dependency_list=None, fov_width=2
) -> (np.ndarray, float, float):
    # from Dyana Beabout

    num_dep, num_field_angles, rsp_func_width = np.shape(response_cube.data)

    dependency_list = [t for (_, t) in response_cube.meta["temperatures"]]
    dependency_list = np.round(dependency_list, decimals=2)
    field_angle_list = [a for (_, a) in response_cube.meta["field_angles"]]
    field_angle_list = np.round(field_angle_list, decimals=2)

    if response_dependency_list is None:
        dep_index_list = [i for (i, _) in response_cube.meta["temperatures"]]
        dep_list_deltas = abs(np.diff(dependency_list))
        max_dep_list_delta = max(dep_list_deltas)
    else:
        dep_list_deltas = abs(np.diff(dependency_list))
        max_dep_list_delta = max(dep_list_deltas)
        dep_index_list = []
        for dep in response_dependency_list:
            delta_dep_list = abs(dependency_list - dep)
            dep_index = np.argmin(delta_dep_list)
            if abs(dependency_list[dep_index] - dep) < max_dep_list_delta:
                dep_index_list = np.append(dep_index_list, dep_index)
        new_index_list = [*set(dep_index_list)]
        new_index_list = np.array(new_index_list, dtype=np.int32)
        new_index_list.sort()
        dep_index_list = new_index_list
        dependency_list = dependency_list[new_index_list]

    num_deps = len(dependency_list)

    field_angle_list_deltas = abs(np.diff(field_angle_list))
    max_field_angle_list_delta = max(field_angle_list_deltas)
    if field_angle_range is None:
        begin_slit_index = np.int64(0)
        end_slit_index = np.int64(len(field_angle_list) - 1)
        field_angle_range_index_list = [begin_slit_index, end_slit_index]
        field_angle_range_list = field_angle_list[field_angle_range_index_list]
    else:
        angle_index_list = []
        for angle in field_angle_range:
            delta_angle_list = abs(field_angle_list - angle)
            angle_index = np.argmin(delta_angle_list)
            if abs(field_angle_list[angle_index] - angle) < max_field_angle_list_delta:
                angle_index_list = np.append(angle_index_list, angle_index)
        new_index_list = [*set(angle_index_list)]
        new_index_list = np.array(new_index_list, dtype=np.int32)
        new_index_list.sort()
        field_angle_range_index_list = new_index_list
        field_angle_range_list = field_angle_list[new_index_list]
        begin_slit_index = field_angle_range_index_list[0]
        end_slit_index = field_angle_range_index_list[1]
        num_field_angles = (end_slit_index - begin_slit_index) + 1

    # Check if number of field angles is even.
    calc_half_fields_angles = divmod(num_field_angles, 2)
    if calc_half_fields_angles[1] == 0.0:
        end_slit_index = end_slit_index - 1
        field_angle_range_index_list[1] = end_slit_index
        field_angle_range_list[1] = field_angle_list[end_slit_index]
        num_field_angles = (end_slit_index - begin_slit_index) + 1

    calc_num_slits = divmod(num_field_angles, fov_width)
    num_slits = int(calc_num_slits[0])
    # Check if number of slits is even.
    calc_half_num_slits = divmod(num_slits, 2)
    if calc_half_num_slits[1] == 0.0:
        num_slits -= 1
    half_slits = divmod(num_slits, 2)

    half_fov = divmod(fov_width, 2)

    center_slit = divmod(end_slit_index - begin_slit_index, 2) + begin_slit_index

    begin_slit_index = center_slit[0] - half_fov[0] - (half_slits[0] * fov_width)
    end_slit_index = center_slit[0] + half_fov[0] + (half_slits[0] * fov_width)

    num_field_angles = (end_slit_index - begin_slit_index) + 1
    field_angle_range_index_list = [begin_slit_index, end_slit_index]
    field_angle_range_list = field_angle_list[field_angle_range_index_list]

    response_count = 0
    response_function = np.zeros((num_deps * num_slits, rsp_func_width), dtype=np.float32)

    for index in dep_index_list:
        # Smooth over dependence.
        slit_count = 0
        for slit_num in range(
            center_slit[0] - (half_slits[0] * fov_width),
            center_slit[0] + ((half_slits[0] * fov_width) + 1),
            fov_width,
        ):
            if fov_width == 1:
                response_function[(num_deps * slit_count) + response_count, :] = response_cube.data[index, slit_num, :]
            else:
                # Check if even FOV.
                if half_fov[1] == 0:
                    response_function[(num_deps * slit_count) + response_count, :] = (
                        response_cube.data[
                            index,
                            slit_num - (half_fov[0] - 1) : slit_num + (half_fov[0] - 1) + 1,
                            :,
                        ].sum(axis=0)
                        + (response_cube.data[index, slit_num - half_fov[0], :] * 0.5)
                        + (response_cube.data[index, slit_num + half_fov[0], :] * 0.5)
                    )
                else:
                    response_function[(num_deps * slit_count) + response_count, :] = response_cube.data[
                        index,
                        slit_num - half_fov[0] : slit_num + half_fov[0] + 1,
                        :,
                    ].sum(axis=0)
            slit_count += 1
        response_count += 1

    return response_function.transpose(), num_slits, num_deps
