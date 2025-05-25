import vtk
import numpy as np

class ParallelVectorsFilter:
    def __init__(self):
        self.first_vector_field_name = None
        self.second_vector_field_name = None

    def set_first_vector_field_name(self, name):
        # Set the name of the first vector field to be used for computation.
        self.first_vector_field_name = name

    def set_second_vector_field_name(self, name):
        # Set the name of the second vector field to be used for computation.
        self.second_vector_field_name = name

    def compute_parallel_vectors(self, dataset):
        # Compute locations where two vector fields are parallel within the input dataset.
        if not self.first_vector_field_name or not self.second_vector_field_name:
            raise ValueError("Vector field names must be set before processing.")

        v_field = dataset.GetPointData().GetArray(self.first_vector_field_name)
        w_field = dataset.GetPointData().GetArray(self.second_vector_field_name)

        if v_field is None or w_field is None:
            raise ValueError("Vector fields not found in the dataset.")

        if v_field.GetNumberOfComponents() != 3 or w_field.GetNumberOfComponents() != 3:
            raise ValueError("Vector fields must have 3 components.")

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        locator = vtk.vtkMergePoints()

        bounds = dataset.GetBounds()
        locator.InitPointInsertion(points, bounds)

        polyline_builder = []

        for cell_id in range(dataset.GetNumberOfCells()):
            cell = dataset.GetCell(cell_id)

            # Process only 3D cells.
            if cell.GetCellDimension() != 3:
                continue

            triangles = self._get_cell_triangles(cell)

            for triangle in triangles:
                v_vectors, w_vectors = self._extract_vectors(triangle, dataset, v_field, w_field)

                # Check if the vector fields are parallel on the triangle.
                st_coords = self._compute_field_alignment_point(v_vectors, w_vectors)

                if st_coords is None:
                    continue

                # Interpolate the location on the triangle where vectors are parallel.
                interpolated_point = self._interpolate_point(st_coords, triangle, dataset)

                if interpolated_point is not None:
                    polyline_builder.append(interpolated_point)

        # Build the output polyline data from collected points.
        self._build_output(points, lines, polyline_builder)

        output = vtk.vtkPolyData()
        output.SetPoints(points)
        output.SetLines(lines)
        return output

    def _get_cell_triangles(self, cell):
        # Generate a list of triangles from the faces of the input cell.
        triangles = []

        for face_id in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(face_id)
            n_points = face.GetNumberOfPoints()

            if n_points < 3:
                continue

            for i in range(1, n_points - 1):
                triangles.append([face.GetPointId(0), face.GetPointId(i), face.GetPointId(i + 1)])

        return triangles

    def _extract_vectors(self, triangle, dataset, v_field, w_field):
        # Extract vector values at the vertices of a triangle.
        v_vectors = []
        w_vectors = []

        for point_id in triangle:
            v_vectors.append(np.array(v_field.GetTuple(point_id)))
            w_vectors.append(np.array(w_field.GetTuple(point_id)))

        return np.array(v_vectors), np.array(w_vectors)

    def _compute_field_alignment_point(self, v_vectors, w_vectors):
        # Compute the parametric location where the two vector fields are parallel.
        V = np.column_stack([v_vectors[1] - v_vectors[0], v_vectors[2] - v_vectors[0], v_vectors[0]])
        W = np.column_stack([w_vectors[1] - w_vectors[0], w_vectors[2] - w_vectors[0], w_vectors[0]])

        # Check for degenerate cases where fields cannot be aligned.
        if np.linalg.det(V) < 1e-12 and np.linalg.det(W) < 1e-12:
            return None

        # Solve for eigenvalues and eigenvectors to find alignment points.
        M = np.linalg.solve(W, V) if np.linalg.det(W) > np.linalg.det(V) else np.linalg.solve(V, W)

        eigenvalues, eigenvectors = np.linalg.eig(M)

        for i in range(len(eigenvalues)):
            if not np.iscomplex(eigenvalues[i]):
                eigenvector = eigenvectors[:, i].real

                if np.abs(eigenvector[2]) > 1e-12:
                    eigenvector /= eigenvector[2]

                    if (eigenvector[0] >= 0 and eigenvector[1] >= 0 and
                            eigenvector[0] + eigenvector[1] <= 1):
                        return eigenvector[:2]

        return None

    def _interpolate_point(self, st_coords, triangle, dataset):
        # Interpolate the point coordinates on the triangle based on parametric values.
        s, t = st_coords

        p_coords = np.array([1 - s - t, s, t])
        point_positions = np.array([dataset.GetPoint(pid) for pid in triangle])

        interpolated_point = np.dot(p_coords, point_positions)
        return interpolated_point

    def _build_output(self, points, lines, polyline_builder):
        # Construct the output polyline from collected points.
        for segment in polyline_builder:
            pid = points.InsertNextPoint(segment)
            lines.InsertNextCell(1)
            lines.InsertCellPoint(pid)


# Usage Example
# data = vtk.vtkUnstructuredGrid()  # Load or create your data here.
# filter = ParallelVectorsFilter()
# filter.set_first_vector_field_name("Velocity")
# filter.set_second_vector_field_name("MagneticField")
# result = filter.compute_parallel_vectors(data)
# # result is vtkPolyData containing the parallel vectors
