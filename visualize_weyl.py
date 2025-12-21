import vtk
import numpy as np
import os
import glob

def load_vtk_file(filename):
    """Load a VTK rectilinear grid file (.vtr)."""
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def create_volume_renderer(image_data, data_range):
    """Create volume renderer for Berry Curvature magnitude (Topological Charge)."""
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(image_data)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    data_min, data_max = data_range
    data_range_size = data_max - data_min
    
    # Opacity transfer function - transparent everywhere except near Weyl points
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(data_min, 0.0)
    opacity_tf.AddPoint(data_min + 0.7 * data_range_size, 0.0)  # Mostly transparent
    opacity_tf.AddPoint(data_min + 0.85 * data_range_size, 0.3)  # Start showing
    opacity_tf.AddPoint(data_min + 0.95 * data_range_size, 0.8)  # Bright near Weyl points
    opacity_tf.AddPoint(data_max, 1.0)  # Maximum at Weyl points
    
    # Color transfer function - Blue to Red (Cold to Hot)
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(data_min, 0.0, 0.0, 0.3)  # Dark blue
    color_tf.AddRGBPoint(data_min + 0.5 * data_range_size, 0.0, 0.5, 1.0)  # Cyan
    color_tf.AddRGBPoint(data_min + 0.8 * data_range_size, 1.0, 1.0, 0.0)  # Yellow
    color_tf.AddRGBPoint(data_max, 1.0, 0.0, 0.0)  # Red (Weyl points)
    
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.SetColor(color_tf)
    
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
    return volume

def create_streamlines(grid_data, num_seeds=100):
    """Create streamlines (hedgehogs) from Berry Curvature vector field."""
    # Create seed points - place them near the center where Weyl points are likely
    seed_points = vtk.vtkPointSource()
    seed_points.SetCenter(0, 0, 0)
    seed_points.SetRadius(2.0)
    seed_points.SetNumberOfPoints(num_seeds)
    seed_points.Update()
    
    # Create stream tracer
    stream_tracer = vtk.vtkStreamTracer()
    stream_tracer.SetInputData(grid_data)
    stream_tracer.SetSourceConnection(seed_points.GetOutputPort())
    stream_tracer.SetMaximumPropagation(10.0)
    stream_tracer.SetInitialIntegrationStep(0.1)
    stream_tracer.SetIntegrationDirectionToBoth()
    
    # Use Runge-Kutta 4 integrator
    rk4 = vtk.vtkRungeKutta4()
    stream_tracer.SetIntegrator(rk4)
    
    # Set vector field - BerryCurvature should be a 3-component vector array
    stream_tracer.SetInputArrayToProcess(0, 0, 0, 
                                        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                        "BerryCurvature")
    
    stream_tracer.Update()
    
    # Create tube filter for better visualization
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(stream_tracer.GetOutputPort())
    tube_filter.SetRadius(0.05)
    tube_filter.SetNumberOfSides(8)
    tube_filter.CappingOn()
    tube_filter.Update()
    
    # Map to colors based on vector magnitude
    magnitude_calc = vtk.vtkArrayCalculator()
    magnitude_calc.SetInputConnection(tube_filter.GetOutputPort())
    magnitude_calc.AddVectorArrayName("BerryCurvature")
    magnitude_calc.SetResultArrayName("Magnitude")
    magnitude_calc.SetFunction("mag(BerryCurvature)")
    magnitude_calc.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(magnitude_calc.GetOutputPort())
    mapper.SetScalarRange(0, 5.0)  # Adjust based on your data range
    mapper.ScalarVisibilityOn()
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.8)
    
    return actor

def create_slice(grid_data, plane_origin, plane_normal, show_contours=True):
    """Create a cross-section slice showing Fermi surface (energy contours)."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_origin)
    plane.SetNormal(plane_normal)
    
    cutter = vtk.vtkCutter()
    cutter.SetInputData(grid_data)
    cutter.SetCutFunction(plane)
    
    if show_contours:
        # Create contours of energy (Fermi surface)
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(cutter.GetOutputPort())
        contour.SetInputArrayToProcess(0, 0, 0, 
                                      vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                      "Energy")
        # Set multiple contour levels to show the Dirac cone structure
        energy_range = grid_data.GetPointData().GetScalars("Energy").GetRange()
        num_contours = 10
        for i in range(num_contours):
            level = energy_range[0] + (energy_range[1] - energy_range[0]) * (i + 1) / (num_contours + 1)
            contour.SetValue(i, level)
        
        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputConnection(contour.GetOutputPort())
        plane_mapper.ScalarVisibilityOn()
        plane_mapper.SetScalarRange(energy_range)
    else:
        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputConnection(cutter.GetOutputPort())
        plane_mapper.ScalarVisibilityOn()
    
    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(plane_mapper)
    plane_actor.GetProperty().SetOpacity(0.6)
    plane_actor.GetProperty().SetLineWidth(2.0)
    
    return plane_actor

def create_weyl_point_markers(grid_data):
    """Mark Weyl points as spheres (where energy is near zero)."""
    # Find points where energy is minimal (Weyl points)
    energy_array = grid_data.GetPointData().GetScalars("Energy")
    if energy_array is None:
        return None
    
    # Threshold filter to find low-energy regions
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid_data)
    threshold.SetInputArrayToProcess(0, 0, 0, 
                                     vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                     "Energy")
    # Some VTK builds lack ThresholdBetween; use lower/upper setters instead.
    threshold.SetLowerThreshold(0.0)
    threshold.SetUpperThreshold(0.5)  # Energy near zero
    threshold.Update()
    
    # Convert to polydata
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(threshold.GetOutputPort())
    
    # Create spheres at each point
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(0.2)
    sphere_source.SetThetaResolution(16)
    sphere_source.SetPhiResolution(16)
    
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputConnection(geometry.GetOutputPort())
    glyph.SetSourceConnection(sphere_source.GetOutputPort())
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetScaleFactor(0.5)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red spheres
    actor.GetProperty().SetOpacity(0.9)
    
    return actor

def main():
    output_dir = "./weyl_output"
    
    # Find all VTK files
    vtk_files = sorted(glob.glob(os.path.join(output_dir, "weyl_M_*.vtr")))
    
    if not vtk_files:
        print(f"Error: No VTK files found in {output_dir}")
        print("Please run generate_weyl_data.py first to generate the data.")
        return
    
    print(f"Found {len(vtk_files)} VTK files")
    print(f"Loading: {os.path.basename(vtk_files[0])}")
    
    # Load first file to get data ranges
    first_data = load_vtk_file(vtk_files[0])
    
    # Get Berry Magnitude range
    berry_mag_array = first_data.GetPointData().GetScalars("BerryMagnitude")
    if berry_mag_array is None:
        print("Error: 'BerryMagnitude' field not found in VTK file")
        return
    
    berry_mag_range = berry_mag_array.GetRange()
    print(f"Berry Curvature Magnitude range: {berry_mag_range}")
    
    # Get Energy range
    energy_array = first_data.GetPointData().GetScalars("Energy")
    if energy_array is None:
        print("Error: 'Energy' field not found in VTK file")
        return
    
    energy_range = energy_array.GetRange()
    print(f"Energy range: {energy_range}")
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
    
    # Load and visualize first frame
    current_frame = 0
    current_data = load_vtk_file(vtk_files[current_frame])
    
    # 1. Volume Rendering (Topological Charge - glowing Weyl points)
    print("Creating volume renderer...")
    volume = create_volume_renderer(current_data, berry_mag_range)
    renderer.AddVolume(volume)
    
    # 2. Streamlines (Hedgehogs - Berry Curvature field lines)
    print("Creating streamlines...")
    streamlines_actor = create_streamlines(current_data, num_seeds=150)
    renderer.AddActor(streamlines_actor)
    
    # 3. Cross-Section (Fermi Surface - Energy contours)
    print("Creating cross-section...")
    slice_actor = create_slice(current_data, [0, 0, 0], [0, 0, 1], show_contours=True)  # Z-normal slice
    renderer.AddActor(slice_actor)
    
    # 4. Weyl Point Markers (Optional - red spheres)
    print("Creating Weyl point markers...")
    weyl_markers = create_weyl_point_markers(current_data)
    if weyl_markers:
        renderer.AddActor(weyl_markers)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1920, 1080)
    render_window.SetWindowName("Weyl Semimetal: Berry Curvature Visualization")
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Set up camera
    camera = renderer.GetActiveCamera()
    camera.SetPosition(8, 8, 8)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    
    # Animation state
    animation_paused = False
    
    # Store actors for updating
    actors = {
        'volume': volume,
        'streamlines': streamlines_actor,
        'slice': slice_actor,
        'weyl_markers': weyl_markers
    }
    
    # Animation callback
    def update_frame(obj, event):
        nonlocal current_frame, current_data, actors, animation_paused
        
        if animation_paused:
            return
        
        # Update to next frame
        current_frame = (current_frame + 1) % len(vtk_files)
        current_data = load_vtk_file(vtk_files[current_frame])
        
        # Update volume
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(current_data)
        actors['volume'].SetMapper(volume_mapper)
        
        # Update streamlines
        renderer.RemoveActor(actors['streamlines'])
        actors['streamlines'] = create_streamlines(current_data, num_seeds=150)
        renderer.AddActor(actors['streamlines'])
        
        # Update slice
        renderer.RemoveActor(actors['slice'])
        actors['slice'] = create_slice(current_data, [0, 0, 0], [0, 0, 1], show_contours=True)
        renderer.AddActor(actors['slice'])
        
        # Update Weyl markers
        if actors['weyl_markers']:
            renderer.RemoveActor(actors['weyl_markers'])
        actors['weyl_markers'] = create_weyl_point_markers(current_data)
        if actors['weyl_markers']:
            renderer.AddActor(actors['weyl_markers'])
        
        # Update window title with frame info
        frame_name = os.path.basename(vtk_files[current_frame])
        render_window.SetWindowName(f"Weyl Semimetal: Berry Curvature - {frame_name} ({current_frame+1}/{len(vtk_files)})")
        
        render_window.Render()
    
    # Timer for animation (update every 300ms)
    timer_id = interactor.CreateRepeatingTimer(300)
    interactor.AddObserver("TimerEvent", update_frame)
    
    # Keyboard controls
    def key_press_callback(obj, event):
        nonlocal current_frame, animation_paused
        key = obj.GetKeySym()
        if key == 'space':
            # Pause/resume animation
            animation_paused = not animation_paused
            print("Animation paused" if animation_paused else "Animation resumed")
        elif key == 'r' or key == 'R':
            # Reset to first frame
            current_frame = 0
            update_frame(None, None)
        elif key == 'n' or key == 'N':
            # Next frame
            animation_paused = True
            update_frame(None, None)
        elif key == 's' or key == 'S':
            # Toggle streamlines
            if actors['streamlines']:
                visible = actors['streamlines'].GetVisibility()
                actors['streamlines'].SetVisibility(not visible)
                render_window.Render()
        elif key == 'v' or key == 'V':
            # Toggle volume rendering
            if actors['volume']:
                visible = actors['volume'].GetVisibility()
                actors['volume'].SetVisibility(not visible)
                render_window.Render()
        elif key == 'c' or key == 'C':
            # Toggle cross-section
            if actors['slice']:
                visible = actors['slice'].GetVisibility()
                actors['slice'].SetVisibility(not visible)
                render_window.Render()
    
    interactor.AddObserver("KeyPressEvent", key_press_callback)
    
    # Render
    render_window.Render()
    
    print("\n" + "=" * 60)
    print("Visualization ready!")
    print("=" * 60)
    print("Controls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Zoom")
    print("  - Middle click + drag: Pan")
    print("  - Space: Pause/Resume animation")
    print("  - R: Reset to first frame")
    print("  - N: Next frame (single step)")
    print("  - S: Toggle streamlines (hedgehogs)")
    print("  - V: Toggle volume rendering (topological charge)")
    print("  - C: Toggle cross-section (Fermi surface)")
    print("  - Close window to exit")
    print("=" * 60)
    
    interactor.Start()

if __name__ == '__main__':
    main()

