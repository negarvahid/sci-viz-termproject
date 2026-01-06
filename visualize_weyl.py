"""
Complete Weyl Semimetal Visualization
=====================================

Features:
1. Cross-sections in any spatial dimension (X, Y, Z planes) with contours
2. Stream functions (streamlines from Berry curvature vector field)
3. Volume rendering - Direct (ray casting) and Indirect (isosurfaces)
4. Ray tracing with visible ray path
5. Animations in time (through mass parameter M)

Controls:
- Space: Pause/Resume animation
- 1/2/3: Toggle X/Y/Z cross-section planes
- S: Toggle streamlines
- V: Toggle direct volume rendering
- I: Toggle indirect volume rendering (isosurfaces)
- R: Toggle ray tracing visualization
- A: Cycle through all ray paths
- +/-: Move cross-section planes
- Arrow keys: Rotate ray direction
- N: Next frame (when paused)
- P: Previous frame (when paused)
- 0: Reset to first frame
- H: Print help
- Q: Quit
"""

import vtk
import numpy as np
import os
import glob
from vtk.util import numpy_support


class WeylVisualizer:
    """Complete visualization system for Weyl semimetal Berry curvature data."""
    
    def __init__(self, output_dir="./weyl_output"):
        self.output_dir = output_dir
        self.vtk_files = []
        self.current_frame = 0
        self.current_data = None
        self.animation_paused = True  # Start paused
        
        # Data ranges (computed from first file)
        self.berry_mag_range = (0, 1)
        self.energy_range = (0, 1)
        self.bounds = [-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi]
        
        # Slice positions (normalized 0-1)
        self.slice_positions = {'X': 0.5, 'Y': 0.5, 'Z': 0.5}
        
        # Visibility flags
        self.visibility = {
            'volume_direct': True,
            'volume_indirect': False,
            'slice_X': False,
            'slice_Y': False,
            'slice_Z': True,
            'streamlines': True,
            'ray_tracing': False,
            'weyl_markers': True,
            'axes': True,
            'colorbar': True
        }
        
        # Ray tracing parameters
        self.ray_origin = np.array([-3.0, -3.0, -3.0])
        self.ray_direction = np.array([1.0, 1.0, 1.0])
        self.ray_paths = self._generate_ray_paths()
        self.current_ray_path = 0
        
        # VTK objects
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
        # Actor storage
        self.actors = {}
        
    def _generate_ray_paths(self):
        """Generate multiple ray paths for demonstration."""
        paths = []
        # Diagonal rays
        paths.append((np.array([-3, -3, -3]), np.array([1, 1, 1])))
        paths.append((np.array([3, -3, -3]), np.array([-1, 1, 1])))
        paths.append((np.array([-3, 3, -3]), np.array([1, -1, 1])))
        paths.append((np.array([-3, -3, 3]), np.array([1, 1, -1])))
        # Axis-aligned rays through origin
        paths.append((np.array([-3, 0, 0]), np.array([1, 0, 0])))
        paths.append((np.array([0, -3, 0]), np.array([0, 1, 0])))
        paths.append((np.array([0, 0, -3]), np.array([0, 0, 1])))
        # Rays through Weyl points (approximately)
        paths.append((np.array([-3, 0, 1]), np.array([1, 0, 0])))
        paths.append((np.array([-3, 0, -1]), np.array([1, 0, 0])))
        return paths
    
    def load_data(self):
        """Load VTK files from output directory."""
        self.vtk_files = sorted(glob.glob(os.path.join(self.output_dir, "weyl_M_*.vtr")))
        
        if not self.vtk_files:
            raise FileNotFoundError(f"No VTK files found in {self.output_dir}")
        
        print(f"Found {len(self.vtk_files)} VTK files")
        
        # Load first file to get data ranges
        self.current_data = self._load_vtk_file(self.vtk_files[0])
        
        # Get data ranges
        berry_mag = self.current_data.GetPointData().GetScalars("BerryMagnitude")
        if berry_mag:
            self.berry_mag_range = berry_mag.GetRange()
            print(f"Berry Magnitude range: {self.berry_mag_range}")
        
        energy = self.current_data.GetPointData().GetScalars("Energy")
        if energy:
            self.energy_range = energy.GetRange()
            print(f"Energy range: {self.energy_range}")
        
        self.bounds = list(self.current_data.GetBounds())
        print(f"Data bounds: {self.bounds}")
        
        return True
    
    def _load_vtk_file(self, filename):
        """Load a single VTK rectilinear grid file."""
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(filename)
        reader.Update()
        return reader.GetOutput()
    
    def setup_renderer(self):
        """Initialize VTK renderer, window, and interactor."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.02, 0.02, 0.08)  # Dark blue background
        self.renderer.SetBackground2(0.0, 0.0, 0.0)  # Gradient to black
        self.renderer.GradientBackgroundOn()
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1920, 1080)
        self.render_window.SetWindowName("Weyl Semimetal: Complete Berry Curvature Visualization")
        
        # Enable anti-aliasing
        self.render_window.SetMultiSamples(4)
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Setup camera
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(10, 10, 10)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
    
    # =========================================================================
    # 1. CROSS-SECTIONS WITH CONTOURS
    # =========================================================================
    
    def create_cross_section(self, axis='Z', show_contours=True, scalar_field="Energy"):
        """
        Create a cross-section plane in any spatial dimension with contours.
        
        Args:
            axis: 'X', 'Y', or 'Z' - normal direction of the plane
            show_contours: Whether to show contour lines
            scalar_field: Field to use for coloring/contouring
        """
        # Calculate plane position from normalized value
        pos = self.slice_positions[axis]
        
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        axis_idx = axis_map[axis]
        
        min_val = self.bounds[axis_idx * 2]
        max_val = self.bounds[axis_idx * 2 + 1]
        plane_pos = min_val + pos * (max_val - min_val)
        
        # Create plane
        plane = vtk.vtkPlane()
        origin = [0, 0, 0]
        normal = [0, 0, 0]
        origin[axis_idx] = plane_pos
        normal[axis_idx] = 1.0
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        
        # Cut the data with the plane
        cutter = vtk.vtkCutter()
        cutter.SetInputData(self.current_data)
        cutter.SetCutFunction(plane)
        cutter.Update()
        
        # Create assembly to hold both slice and contours
        assembly = vtk.vtkAssembly()
        
        # Colored slice
        slice_mapper = vtk.vtkPolyDataMapper()
        slice_mapper.SetInputConnection(cutter.GetOutputPort())
        slice_mapper.SetScalarModeToUsePointFieldData()
        slice_mapper.SelectColorArray(scalar_field)
        
        if scalar_field == "Energy":
            slice_mapper.SetScalarRange(self.energy_range)
        else:
            slice_mapper.SetScalarRange(self.berry_mag_range)
        
        slice_actor = vtk.vtkActor()
        slice_actor.SetMapper(slice_mapper)
        slice_actor.GetProperty().SetOpacity(0.7)
        assembly.AddPart(slice_actor)
        
        # Contour lines on the slice
        if show_contours:
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(cutter.GetOutputPort())
            contour.SetInputArrayToProcess(0, 0, 0,
                                          vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                          scalar_field)
            
            # Generate contour levels
            if scalar_field == "Energy":
                data_range = self.energy_range
            else:
                data_range = self.berry_mag_range
            
            num_contours = 15
            for i in range(num_contours):
                level = data_range[0] + (data_range[1] - data_range[0]) * (i + 1) / (num_contours + 1)
                contour.SetValue(i, level)
            
            contour.Update()
            
            # Tube filter for better visibility of contour lines
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(contour.GetOutputPort())
            tube.SetRadius(0.02)
            tube.SetNumberOfSides(8)
            tube.Update()
            
            contour_mapper = vtk.vtkPolyDataMapper()
            contour_mapper.SetInputConnection(tube.GetOutputPort())
            contour_mapper.SetScalarRange(data_range)
            
            contour_actor = vtk.vtkActor()
            contour_actor.SetMapper(contour_mapper)
            contour_actor.GetProperty().SetLineWidth(2.0)
            assembly.AddPart(contour_actor)
        
        return assembly
    
    # =========================================================================
    # 2. STREAMLINES (STREAM FUNCTIONS)
    # =========================================================================
    
    def create_streamlines(self, num_seeds=200, integration_length=15.0):
        """
        Create streamlines from Berry Curvature vector field.
        
        These represent the "flow" of the Berry curvature field,
        showing how magnetic monopoles (Weyl points) act as sources/sinks.
        """
        # Create seed points distributed throughout the volume
        # Use multiple seeding strategies for better coverage
        
        append_seeds = vtk.vtkAppendPolyData()
        
        # Strategy 1: Random points in volume
        random_seeds = vtk.vtkPointSource()
        random_seeds.SetCenter(0, 0, 0)
        random_seeds.SetRadius(2.5)
        random_seeds.SetNumberOfPoints(num_seeds // 2)
        random_seeds.Update()
        append_seeds.AddInputConnection(random_seeds.GetOutputPort())
        
        # Strategy 2: Points near z-axis (where Weyl points typically are)
        line_seeds = vtk.vtkLineSource()
        line_seeds.SetPoint1(0, 0, -2.5)
        line_seeds.SetPoint2(0, 0, 2.5)
        line_seeds.SetResolution(num_seeds // 4)
        line_seeds.Update()
        append_seeds.AddInputConnection(line_seeds.GetOutputPort())
        
        # Strategy 3: Points on a sphere around origin
        sphere_seeds = vtk.vtkPointSource()
        sphere_seeds.SetCenter(0, 0, 0)
        sphere_seeds.SetRadius(1.0)
        sphere_seeds.SetDistributionToShell()
        sphere_seeds.SetNumberOfPoints(num_seeds // 4)
        sphere_seeds.Update()
        append_seeds.AddInputConnection(sphere_seeds.GetOutputPort())
        
        append_seeds.Update()
        
        # Create stream tracer
        stream_tracer = vtk.vtkStreamTracer()
        stream_tracer.SetInputData(self.current_data)
        stream_tracer.SetSourceConnection(append_seeds.GetOutputPort())
        stream_tracer.SetMaximumPropagation(integration_length)
        stream_tracer.SetInitialIntegrationStep(0.05)
        stream_tracer.SetMaximumIntegrationStep(0.2)
        stream_tracer.SetIntegrationDirectionToBoth()
        stream_tracer.SetComputeVorticity(True)
        
        # Use Runge-Kutta 4-5 integrator for accuracy
        rk45 = vtk.vtkRungeKutta45()
        stream_tracer.SetIntegrator(rk45)
        
        # Set the vector field
        stream_tracer.SetInputArrayToProcess(0, 0, 0,
                                            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                            "BerryCurvature")
        stream_tracer.Update()
        
        # Create tubes around streamlines for better visualization
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(stream_tracer.GetOutputPort())
        tube_filter.SetRadius(0.03)
        tube_filter.SetNumberOfSides(8)
        tube_filter.SetVaryRadiusToVaryRadiusByScalar()
        tube_filter.CappingOn()
        tube_filter.Update()
        
        # Color by Berry curvature magnitude along the streamline
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("BerryMagnitude")
        mapper.SetScalarRange(self.berry_mag_range)
        
        # Create a nice colormap
        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(self.berry_mag_range[0], 0.0, 0.0, 0.5)  # Dark blue
        lut.AddRGBPoint(self.berry_mag_range[0] + 0.25 * (self.berry_mag_range[1] - self.berry_mag_range[0]), 0.0, 0.5, 1.0)  # Cyan
        lut.AddRGBPoint(self.berry_mag_range[0] + 0.5 * (self.berry_mag_range[1] - self.berry_mag_range[0]), 0.0, 1.0, 0.0)  # Green
        lut.AddRGBPoint(self.berry_mag_range[0] + 0.75 * (self.berry_mag_range[1] - self.berry_mag_range[0]), 1.0, 1.0, 0.0)  # Yellow
        lut.AddRGBPoint(self.berry_mag_range[1], 1.0, 0.0, 0.0)  # Red
        mapper.SetLookupTable(lut)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.9)
        
        return actor
    
    # =========================================================================
    # 3a. DIRECT VOLUME RENDERING (Ray Casting)
    # =========================================================================
    
    def create_direct_volume_rendering(self):
        """
        Create direct volume rendering using GPU ray casting.
        
        This renders the Berry curvature magnitude as a semi-transparent volume,
        with high values (near Weyl points) appearing bright.
        """
        # Use GPU ray cast mapper for direct volume rendering
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(self.current_data)
        volume_mapper.SetScalarModeToUsePointFieldData()
        volume_mapper.SelectScalarArray("BerryMagnitude")
        
        # Ray casting parameters
        volume_mapper.SetSampleDistance(0.05)
        volume_mapper.SetAutoAdjustSampleDistances(True)
        volume_mapper.SetBlendModeToComposite()
        
        # Volume properties
        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Lighting for depth perception
        volume_property.SetAmbient(0.3)
        volume_property.SetDiffuse(0.7)
        volume_property.SetSpecular(0.4)
        volume_property.SetSpecularPower(20.0)
        
        # Opacity transfer function - mostly transparent, opaque near Weyl points
        opacity_tf = vtk.vtkPiecewiseFunction()
        data_min, data_max = self.berry_mag_range
        data_range = data_max - data_min
        
        opacity_tf.AddPoint(data_min, 0.0)
        opacity_tf.AddPoint(data_min + 0.5 * data_range, 0.0)
        opacity_tf.AddPoint(data_min + 0.7 * data_range, 0.05)
        opacity_tf.AddPoint(data_min + 0.85 * data_range, 0.2)
        opacity_tf.AddPoint(data_min + 0.95 * data_range, 0.6)
        opacity_tf.AddPoint(data_max, 1.0)
        
        # Color transfer function - blue to red (cold to hot)
        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(data_min, 0.0, 0.0, 0.2)
        color_tf.AddRGBPoint(data_min + 0.25 * data_range, 0.0, 0.0, 0.8)
        color_tf.AddRGBPoint(data_min + 0.5 * data_range, 0.0, 0.8, 0.8)
        color_tf.AddRGBPoint(data_min + 0.75 * data_range, 0.8, 0.8, 0.0)
        color_tf.AddRGBPoint(data_max, 1.0, 0.0, 0.0)
        
        # Gradient opacity for edge enhancement
        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_opacity.AddPoint(0, 0.0)
        gradient_opacity.AddPoint(0.5, 0.5)
        gradient_opacity.AddPoint(1.0, 1.0)
        
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.SetColor(color_tf)
        volume_property.SetGradientOpacity(gradient_opacity)
        
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        return volume
    
    # =========================================================================
    # 3b. INDIRECT VOLUME RENDERING (Isosurfaces)
    # =========================================================================
    
    def create_indirect_volume_rendering(self, num_isosurfaces=5):
        """
        Create indirect volume rendering using isosurfaces (marching cubes).
        
        This extracts surfaces of constant Berry curvature magnitude,
        providing a different view of the field structure.
        """
        assembly = vtk.vtkAssembly()
        
        data_min, data_max = self.berry_mag_range
        data_range = data_max - data_min
        
        # Create lookup table for coloring
        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(data_min, 0.0, 0.0, 0.8)
        lut.AddRGBPoint(data_min + 0.5 * data_range, 0.0, 0.8, 0.8)
        lut.AddRGBPoint(data_max, 1.0, 0.0, 0.0)
        
        # Generate isosurfaces at different values
        for i in range(num_isosurfaces):
            # Use values concentrated near the high end (where Weyl points are)
            t = (i + 1) / (num_isosurfaces + 1)
            t = t ** 0.5  # Bias toward higher values
            iso_value = data_min + 0.5 * data_range + t * 0.5 * data_range
            
            # Marching cubes for isosurface extraction
            contour = vtk.vtkContourFilter()
            contour.SetInputData(self.current_data)
            contour.SetInputArrayToProcess(0, 0, 0,
                                          vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                          "BerryMagnitude")
            contour.SetValue(0, iso_value)
            contour.ComputeNormalsOn()
            contour.Update()
            
            # Smooth the isosurface
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(0.001)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Color and opacity based on isosurface value
            color = [0, 0, 0]
            lut.GetColor(iso_value, color)
            actor.GetProperty().SetColor(color)
            
            # Outer surfaces more transparent
            opacity = 0.2 + 0.6 * t
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(20)
            
            assembly.AddPart(actor)
        
        return assembly
    
    # =========================================================================
    # 4. RAY TRACING WITH VISIBLE PATH
    # =========================================================================
    
    def create_ray_tracing_visualization(self):
        """
        Create visualization of ray tracing through the volume.
        
        Shows:
        - The ray path as a line/tube
        - Sample points along the ray
        - Accumulated values at each sample point
        - Hit points where the ray intersects isosurfaces
        """
        assembly = vtk.vtkAssembly()
        
        origin = self.ray_paths[self.current_ray_path][0]
        direction = self.ray_paths[self.current_ray_path][1]
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Calculate ray end point (through the volume)
        ray_length = 10.0
        end_point = origin + direction * ray_length
        
        # 1. Ray path as a glowing tube
        ray_line = vtk.vtkLineSource()
        ray_line.SetPoint1(origin)
        ray_line.SetPoint2(end_point)
        ray_line.SetResolution(100)
        ray_line.Update()
        
        ray_tube = vtk.vtkTubeFilter()
        ray_tube.SetInputConnection(ray_line.GetOutputPort())
        ray_tube.SetRadius(0.05)
        ray_tube.SetNumberOfSides(16)
        ray_tube.CappingOn()
        ray_tube.Update()
        
        ray_mapper = vtk.vtkPolyDataMapper()
        ray_mapper.SetInputConnection(ray_tube.GetOutputPort())
        
        ray_actor = vtk.vtkActor()
        ray_actor.SetMapper(ray_mapper)
        ray_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow ray
        ray_actor.GetProperty().SetOpacity(0.8)
        ray_actor.GetProperty().SetAmbient(0.5)
        assembly.AddPart(ray_actor)
        
        # 2. Sample points along the ray
        num_samples = 50
        sample_points = vtk.vtkPoints()
        sample_values = vtk.vtkFloatArray()
        sample_values.SetName("SampleValue")
        
        # Get the data array for sampling
        berry_mag = self.current_data.GetPointData().GetScalars("BerryMagnitude")
        
        # Create a probe filter to sample the volume along the ray
        probe_points = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            point = origin + direction * ray_length * t
            points.InsertNextPoint(point)
        
        probe_points.SetPoints(points)
        
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(probe_points)
        probe.SetSourceData(self.current_data)
        probe.Update()
        
        # Get sampled values
        probed_data = probe.GetOutput()
        probed_values = probed_data.GetPointData().GetArray("BerryMagnitude")
        
        # 3. Create spheres at sample points, sized by sampled value
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(1.0)
        sphere_source.SetThetaResolution(12)
        sphere_source.SetPhiResolution(12)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(probed_data)
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(0.15)
        glyph.SetColorModeToColorByScalar()
        
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyph.GetOutputPort())
        glyph_mapper.SetScalarRange(self.berry_mag_range)
        
        # Color map for sample points
        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(self.berry_mag_range[0], 0.0, 0.0, 1.0)  # Blue (low)
        lut.AddRGBPoint(self.berry_mag_range[1], 1.0, 0.0, 0.0)  # Red (high)
        glyph_mapper.SetLookupTable(lut)
        
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetOpacity(0.9)
        assembly.AddPart(glyph_actor)
        
        # 4. Ray origin marker (large sphere)
        origin_sphere = vtk.vtkSphereSource()
        origin_sphere.SetCenter(origin)
        origin_sphere.SetRadius(0.15)
        origin_sphere.SetThetaResolution(20)
        origin_sphere.SetPhiResolution(20)
        
        origin_mapper = vtk.vtkPolyDataMapper()
        origin_mapper.SetInputConnection(origin_sphere.GetOutputPort())
        
        origin_actor = vtk.vtkActor()
        origin_actor.SetMapper(origin_mapper)
        origin_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
        origin_actor.GetProperty().SetOpacity(1.0)
        assembly.AddPart(origin_actor)
        
        # 5. Arrow showing ray direction
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(20)
        arrow.SetShaftResolution(20)
        
        arrow_transform = vtk.vtkTransform()
        arrow_transform.Translate(origin)
        
        # Calculate rotation to align arrow with ray direction
        default_dir = np.array([1, 0, 0])
        rotation_axis = np.cross(default_dir, direction)
        rotation_angle = np.arccos(np.clip(np.dot(default_dir, direction), -1, 1))
        
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            arrow_transform.RotateWXYZ(np.degrees(rotation_angle),
                                       rotation_axis[0], rotation_axis[1], rotation_axis[2])
        
        arrow_transform.Scale(0.8, 0.8, 0.8)
        
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(arrow.GetOutputPort())
        transform_filter.SetTransform(arrow_transform)
        
        arrow_mapper = vtk.vtkPolyDataMapper()
        arrow_mapper.SetInputConnection(transform_filter.GetOutputPort())
        
        arrow_actor = vtk.vtkActor()
        arrow_actor.SetMapper(arrow_mapper)
        arrow_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
        assembly.AddPart(arrow_actor)
        
        # 6. Text annotation showing ray info
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Ray {self.current_ray_path + 1}/{len(self.ray_paths)}\n"
                           f"Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})\n"
                           f"Dir: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)
        text_actor.SetPosition(10, 10)
        
        return assembly, text_actor
    
    # =========================================================================
    # 5. WEYL POINT MARKERS
    # =========================================================================
    
    def create_weyl_markers(self):
        """Create markers at Weyl points (where energy approaches zero)."""
        # Threshold to find low-energy regions
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(self.current_data)
        threshold.SetInputArrayToProcess(0, 0, 0,
                                        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                        "Energy")
        threshold.SetLowerThreshold(0.0)
        threshold.SetUpperThreshold(0.3)
        threshold.Update()
        
        # Convert to polydata
        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputConnection(threshold.GetOutputPort())
        geometry.Update()
        
        # Create glyphs at Weyl points
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.15)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(geometry.GetOutputPort())
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.5)  # Magenta
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetSpecular(0.8)
        actor.GetProperty().SetSpecularPower(30)
        
        return actor
    
    # =========================================================================
    # AUXILIARY VISUALIZATIONS
    # =========================================================================
    
    def create_axes(self):
        """Create coordinate axes."""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(3.5, 3.5, 3.5)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        axes.SetConeRadius(0.1)
        
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        
        # Label as momentum space
        axes.SetXAxisLabelText("kx")
        axes.SetYAxisLabelText("ky")
        axes.SetZAxisLabelText("kz")
        
        return axes
    
    def create_colorbar(self, title="Berry Curvature\nMagnitude"):
        """Create a color bar legend."""
        lut = vtk.vtkColorTransferFunction()
        data_min, data_max = self.berry_mag_range
        data_range = data_max - data_min
        
        lut.AddRGBPoint(data_min, 0.0, 0.0, 0.2)
        lut.AddRGBPoint(data_min + 0.25 * data_range, 0.0, 0.0, 0.8)
        lut.AddRGBPoint(data_min + 0.5 * data_range, 0.0, 0.8, 0.8)
        lut.AddRGBPoint(data_min + 0.75 * data_range, 0.8, 0.8, 0.0)
        lut.AddRGBPoint(data_max, 1.0, 0.0, 0.0)
        
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle(title)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.9, 0.1)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.8)
        
        scalar_bar.GetTitleTextProperty().SetFontSize(12)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetLabelTextProperty().SetFontSize(10)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        
        return scalar_bar
    
    def create_bounding_box(self):
        """Create a wireframe bounding box showing the Brillouin zone."""
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(self.current_data)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        actor.GetProperty().SetLineWidth(1.5)
        
        return actor
    
    def create_info_text(self):
        """Create text overlay with current frame info."""
        text_actor = vtk.vtkTextActor()
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        text_actor.SetPosition(10, 50)
        return text_actor
    
    # =========================================================================
    # SCENE MANAGEMENT
    # =========================================================================
    
    def build_scene(self):
        """Build the complete visualization scene."""
        print("\nBuilding visualization scene...")
        
        # Clear existing actors
        self.renderer.RemoveAllViewProps()
        self.actors.clear()
        
        # Bounding box
        self.actors['bbox'] = self.create_bounding_box()
        self.renderer.AddActor(self.actors['bbox'])
        
        # Direct volume rendering
        print("  Creating direct volume rendering...")
        self.actors['volume_direct'] = self.create_direct_volume_rendering()
        self.actors['volume_direct'].SetVisibility(self.visibility['volume_direct'])
        self.renderer.AddVolume(self.actors['volume_direct'])
        
        # Indirect volume rendering (isosurfaces)
        print("  Creating indirect volume rendering...")
        self.actors['volume_indirect'] = self.create_indirect_volume_rendering()
        self.actors['volume_indirect'].SetVisibility(self.visibility['volume_indirect'])
        self.renderer.AddActor(self.actors['volume_indirect'])
        
        # Cross-sections
        print("  Creating cross-sections...")
        for axis in ['X', 'Y', 'Z']:
            self.actors[f'slice_{axis}'] = self.create_cross_section(axis)
            self.actors[f'slice_{axis}'].SetVisibility(self.visibility[f'slice_{axis}'])
            self.renderer.AddActor(self.actors[f'slice_{axis}'])
        
        # Streamlines
        print("  Creating streamlines...")
        self.actors['streamlines'] = self.create_streamlines()
        self.actors['streamlines'].SetVisibility(self.visibility['streamlines'])
        self.renderer.AddActor(self.actors['streamlines'])
        
        # Ray tracing visualization
        print("  Creating ray tracing visualization...")
        ray_assembly, ray_text = self.create_ray_tracing_visualization()
        self.actors['ray_tracing'] = ray_assembly
        self.actors['ray_text'] = ray_text
        self.actors['ray_tracing'].SetVisibility(self.visibility['ray_tracing'])
        self.actors['ray_text'].SetVisibility(self.visibility['ray_tracing'])
        self.renderer.AddActor(self.actors['ray_tracing'])
        self.renderer.AddActor2D(self.actors['ray_text'])
        
        # Weyl point markers
        print("  Creating Weyl point markers...")
        self.actors['weyl_markers'] = self.create_weyl_markers()
        self.actors['weyl_markers'].SetVisibility(self.visibility['weyl_markers'])
        self.renderer.AddActor(self.actors['weyl_markers'])
        
        # Axes
        self.actors['axes'] = self.create_axes()
        self.actors['axes'].SetVisibility(self.visibility['axes'])
        self.renderer.AddActor(self.actors['axes'])
        
        # Colorbar
        self.actors['colorbar'] = self.create_colorbar()
        self.actors['colorbar'].SetVisibility(self.visibility['colorbar'])
        self.renderer.AddActor2D(self.actors['colorbar'])
        
        # Info text
        self.actors['info_text'] = self.create_info_text()
        self.renderer.AddActor2D(self.actors['info_text'])
        
        self.update_info_text()
        self.renderer.ResetCamera()
        
        print("Scene built successfully!")
    
    def update_info_text(self):
        """Update the info text overlay."""
        frame_name = os.path.basename(self.vtk_files[self.current_frame])
        status = "PAUSED" if self.animation_paused else "PLAYING"
        
        info = (f"Frame: {self.current_frame + 1}/{len(self.vtk_files)} [{status}]\n"
                f"File: {frame_name}\n"
                f"Press H for help")
        
        self.actors['info_text'].SetInput(info)
    
    def update_frame(self, frame_delta=1):
        """Update to a new frame."""
        self.current_frame = (self.current_frame + frame_delta) % len(self.vtk_files)
        self.current_data = self._load_vtk_file(self.vtk_files[self.current_frame])
        
        # Rebuild scene with new data
        self.build_scene()
        self.render_window.Render()
    
    def update_slice_position(self, axis, delta):
        """Update cross-section position."""
        self.slice_positions[axis] = np.clip(self.slice_positions[axis] + delta, 0.0, 1.0)
        
        # Rebuild the specific slice
        self.renderer.RemoveActor(self.actors[f'slice_{axis}'])
        self.actors[f'slice_{axis}'] = self.create_cross_section(axis)
        self.actors[f'slice_{axis}'].SetVisibility(self.visibility[f'slice_{axis}'])
        self.renderer.AddActor(self.actors[f'slice_{axis}'])
        
        self.render_window.Render()
    
    def cycle_ray_path(self):
        """Cycle through predefined ray paths."""
        self.current_ray_path = (self.current_ray_path + 1) % len(self.ray_paths)
        
        # Rebuild ray visualization
        self.renderer.RemoveActor(self.actors['ray_tracing'])
        self.renderer.RemoveActor2D(self.actors['ray_text'])
        
        ray_assembly, ray_text = self.create_ray_tracing_visualization()
        self.actors['ray_tracing'] = ray_assembly
        self.actors['ray_text'] = ray_text
        self.actors['ray_tracing'].SetVisibility(self.visibility['ray_tracing'])
        self.actors['ray_text'].SetVisibility(self.visibility['ray_tracing'])
        self.renderer.AddActor(self.actors['ray_tracing'])
        self.renderer.AddActor2D(self.actors['ray_text'])
        
        self.render_window.Render()
    
    def toggle_visibility(self, key):
        """Toggle visibility of a scene element."""
        if key in self.visibility:
            self.visibility[key] = not self.visibility[key]
            
            if key in self.actors:
                self.actors[key].SetVisibility(self.visibility[key])
            
            # Handle ray text separately
            if key == 'ray_tracing' and 'ray_text' in self.actors:
                self.actors['ray_text'].SetVisibility(self.visibility[key])
            
            self.render_window.Render()
            print(f"{key}: {'ON' if self.visibility[key] else 'OFF'}")
    
    def print_help(self):
        """Print help information."""
        help_text = """
================================================================================
                    WEYL SEMIMETAL VISUALIZATION - CONTROLS
================================================================================

ANIMATION:
  Space         Pause/Resume animation
  N             Next frame (when paused)
  P             Previous frame (when paused)
  0             Reset to first frame

VISUALIZATION TOGGLES:
  V             Toggle direct volume rendering (ray casting)
  I             Toggle indirect volume rendering (isosurfaces)
  S             Toggle streamlines
  1             Toggle X cross-section
  2             Toggle Y cross-section
  3             Toggle Z cross-section
  R             Toggle ray tracing visualization
  A             Cycle through ray paths
  W             Toggle Weyl point markers

CROSS-SECTION POSITION:
  +/=           Move active cross-section forward
  -/_           Move active cross-section backward
  X/Y/Z         Set active cross-section axis

CAMERA:
  Left-click    Rotate
  Right-click   Zoom
  Middle-click  Pan

OTHER:
  H             Show this help
  Q             Quit

================================================================================
"""
        print(help_text)
    
    # =========================================================================
    # CALLBACKS AND MAIN LOOP
    # =========================================================================
    
    def setup_callbacks(self):
        """Set up keyboard and timer callbacks."""
        self.active_axis = 'Z'  # Active axis for cross-section movement
        
        def key_press_callback(obj, event):
            key = obj.GetKeySym().lower()
            
            if key == 'space':
                self.animation_paused = not self.animation_paused
                self.update_info_text()
                self.render_window.Render()
                print("Animation", "PAUSED" if self.animation_paused else "PLAYING")
            
            elif key == 'n':
                self.animation_paused = True
                self.update_frame(1)
            
            elif key == 'p':
                self.animation_paused = True
                self.update_frame(-1)
            
            elif key == '0':
                self.current_frame = 0
                self.update_frame(0)
            
            elif key == 'v':
                self.toggle_visibility('volume_direct')
            
            elif key == 'i':
                self.toggle_visibility('volume_indirect')
            
            elif key == 's':
                self.toggle_visibility('streamlines')
            
            elif key == '1':
                self.toggle_visibility('slice_X')
                self.active_axis = 'X'
            
            elif key == '2':
                self.toggle_visibility('slice_Y')
                self.active_axis = 'Y'
            
            elif key == '3':
                self.toggle_visibility('slice_Z')
                self.active_axis = 'Z'
            
            elif key == 'r':
                self.toggle_visibility('ray_tracing')
            
            elif key == 'a':
                self.cycle_ray_path()
            
            elif key == 'w':
                self.toggle_visibility('weyl_markers')
            
            elif key in ['plus', 'equal']:
                self.update_slice_position(self.active_axis, 0.05)
            
            elif key in ['minus', 'underscore']:
                self.update_slice_position(self.active_axis, -0.05)
            
            elif key == 'x':
                self.active_axis = 'X'
                print(f"Active axis: {self.active_axis}")
            
            elif key == 'y':
                self.active_axis = 'Y'
                print(f"Active axis: {self.active_axis}")
            
            elif key == 'z':
                self.active_axis = 'Z'
                print(f"Active axis: {self.active_axis}")
            
            elif key == 'h':
                self.print_help()
            
            elif key == 'q':
                self.render_window.Finalize()
                self.interactor.TerminateApp()
        
        self.interactor.AddObserver("KeyPressEvent", key_press_callback)
        
        # Timer for animation
        def timer_callback(obj, event):
            if not self.animation_paused:
                self.update_frame(1)
        
        self.interactor.CreateRepeatingTimer(500)  # 500ms between frames
        self.interactor.AddObserver("TimerEvent", timer_callback)
    
    def run(self):
        """Main entry point to run the visualization."""
        print("=" * 70)
        print("         WEYL SEMIMETAL BERRY CURVATURE VISUALIZATION")
        print("=" * 70)
        
        # Load data
        print("\nLoading data...")
        self.load_data()
        
        # Setup renderer
        print("\nSetting up renderer...")
        self.setup_renderer()
        
        # Build scene
        self.build_scene()
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Print help
        self.print_help()
        
        # Start
        print("\nStarting visualization...")
        self.render_window.Render()
        self.interactor.Start()


def main():
    visualizer = WeylVisualizer(output_dir="./weyl_output")
    visualizer.run()


if __name__ == '__main__':
    main()