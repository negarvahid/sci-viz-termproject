"""
Weyl Semimetal Visualization with Video Export
Based on original visualization script - adds MP4/OGV output
"""

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
    """Create volume renderer with explicit GPU ray casting for Berry Curvature magnitude."""
    try:
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        print("Using GPU Volume Ray Cast Mapper")
    except AttributeError:
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetRequestedRenderModeToGPU()
        print("Using SmartVolumeMapper with GPU preference")
    
    volume_mapper.SetInputData(image_data)
    
    if hasattr(volume_mapper, 'SetSampleDistance'):
        volume_mapper.SetSampleDistance(0.1)
    if hasattr(volume_mapper, 'SetAutoAdjustSampleDistances'):
        volume_mapper.SetAutoAdjustSampleDistances(1)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetAmbient(0.2)
    volume_property.SetDiffuse(0.8)
    volume_property.SetSpecular(0.3)
    volume_property.SetSpecularPower(10.0)
    
    data_min, data_max = data_range
    data_range_size = data_max - data_min
    
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(data_min, 0.0)
    opacity_tf.AddPoint(data_min + 0.7 * data_range_size, 0.0)
    opacity_tf.AddPoint(data_min + 0.85 * data_range_size, 0.3)
    opacity_tf.AddPoint(data_min + 0.95 * data_range_size, 0.8)
    opacity_tf.AddPoint(data_max, 1.0)
    
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(data_min, 0.0, 0.0, 0.3)
    color_tf.AddRGBPoint(data_min + 0.5 * data_range_size, 0.0, 0.5, 1.0)
    color_tf.AddRGBPoint(data_min + 0.8 * data_range_size, 1.0, 1.0, 0.0)
    color_tf.AddRGBPoint(data_max, 1.0, 0.0, 0.0)
    
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.SetColor(color_tf)
    
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
    return volume

def create_streamlines(grid_data, num_seeds=100):
    """Create streamlines from Berry Curvature vector field."""
    seed_points = vtk.vtkPointSource()
    seed_points.SetCenter(0, 0, 0)
    seed_points.SetRadius(2.0)
    seed_points.SetNumberOfPoints(num_seeds)
    seed_points.Update()
    
    stream_tracer = vtk.vtkStreamTracer()
    stream_tracer.SetInputData(grid_data)
    stream_tracer.SetSourceConnection(seed_points.GetOutputPort())
    stream_tracer.SetMaximumPropagation(10.0)
    stream_tracer.SetInitialIntegrationStep(0.1)
    stream_tracer.SetIntegrationDirectionToBoth()
    
    rk4 = vtk.vtkRungeKutta4()
    stream_tracer.SetIntegrator(rk4)
    stream_tracer.SetInputArrayToProcess(0, 0, 0, 
                                        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                        "BerryCurvature")
    stream_tracer.Update()
    
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(stream_tracer.GetOutputPort())
    tube_filter.SetRadius(0.05)
    tube_filter.SetNumberOfSides(8)
    tube_filter.CappingOn()
    tube_filter.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube_filter.GetOutputPort())
    mapper.SetScalarRange(0, 5.0)
    mapper.ScalarVisibilityOn()
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.8)
    
    return actor

def create_slice(grid_data, plane_origin, plane_normal, show_contours=True):
    """Create a cross-section slice showing energy contours."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_origin)
    plane.SetNormal(plane_normal)
    
    cutter = vtk.vtkCutter()
    cutter.SetInputData(grid_data)
    cutter.SetCutFunction(plane)
    
    if show_contours:
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(cutter.GetOutputPort())
        contour.SetInputArrayToProcess(0, 0, 0, 
                                      vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                      "Energy")
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
    energy_array = grid_data.GetPointData().GetScalars("Energy")
    if energy_array is None:
        return None
    
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid_data)
    threshold.SetInputArrayToProcess(0, 0, 0, 
                                     vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 
                                     "Energy")
    threshold.SetLowerThreshold(0.0)
    threshold.SetUpperThreshold(0.5)
    threshold.Update()
    
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(threshold.GetOutputPort())
    
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
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    actor.GetProperty().SetOpacity(0.9)
    
    return actor

def create_isosurfaces(grid_data, berry_mag_range, num_surfaces=3):
    """Create indirect volume rendering via isosurfaces."""
    assembly = vtk.vtkAssembly()
    
    data_min, data_max = berry_mag_range
    data_range = data_max - data_min
    
    for i in range(num_surfaces):
        t = (i + 1) / (num_surfaces + 1)
        iso_value = data_min + 0.5 * data_range + t * 0.4 * data_range
        
        contour = vtk.vtkContourFilter()
        contour.SetInputData(grid_data)
        contour.SetInputArrayToProcess(0, 0, 0,
                                      vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                      "BerryMagnitude")
        contour.SetValue(0, iso_value)
        contour.ComputeNormalsOn()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.2 + 0.8*t, 0.5*(1-t), 1.0 - 0.8*t)
        actor.GetProperty().SetOpacity(0.15 + 0.2*t)
        actor.GetProperty().SetSpecular(0.5)
        
        assembly.AddPart(actor)
    
    return assembly

def create_ray_visualization(grid_data, ray_origin, ray_direction, berry_mag_range):
    """Create visible ray path through the volume."""
    assembly = vtk.vtkAssembly()
    
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    ray_length = 8.0
    end_point = ray_origin + ray_direction * ray_length
    
    # Ray line as tube
    ray_line = vtk.vtkLineSource()
    ray_line.SetPoint1(ray_origin)
    ray_line.SetPoint2(end_point)
    ray_line.SetResolution(50)
    
    ray_tube = vtk.vtkTubeFilter()
    ray_tube.SetInputConnection(ray_line.GetOutputPort())
    ray_tube.SetRadius(0.04)
    ray_tube.SetNumberOfSides(12)
    ray_tube.CappingOn()
    
    ray_mapper = vtk.vtkPolyDataMapper()
    ray_mapper.SetInputConnection(ray_tube.GetOutputPort())
    
    ray_actor = vtk.vtkActor()
    ray_actor.SetMapper(ray_mapper)
    ray_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
    ray_actor.GetProperty().SetOpacity(0.9)
    assembly.AddPart(ray_actor)
    
    # Sample points along ray
    num_samples = 30
    probe_points = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        point = ray_origin + ray_direction * ray_length * t
        points.InsertNextPoint(point)
    
    probe_points.SetPoints(points)
    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(probe_points)
    probe.SetSourceData(grid_data)
    probe.Update()
    
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(10)
    sphere.SetPhiResolution(10)
    
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(probe.GetOutput())
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetScaleFactor(0.1)
    
    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())
    glyph_mapper.SetScalarRange(berry_mag_range)
    
    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    assembly.AddPart(glyph_actor)
    
    # Origin marker
    origin_sphere = vtk.vtkSphereSource()
    origin_sphere.SetCenter(ray_origin)
    origin_sphere.SetRadius(0.15)
    
    origin_mapper = vtk.vtkPolyDataMapper()
    origin_mapper.SetInputConnection(origin_sphere.GetOutputPort())
    
    origin_actor = vtk.vtkActor()
    origin_actor.SetMapper(origin_mapper)
    origin_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
    assembly.AddPart(origin_actor)
    
    return assembly

def main():
    output_dir = "./weyl_output"
    
    vtk_files = sorted(glob.glob(os.path.join(output_dir, "weyl_M_*.vtr")))
    
    if not vtk_files:
        print(f"Error: No VTK files found in {output_dir}")
        print("Please run weyl_berry_curvature.py first to generate the data.")
        return
    
    print(f"Found {len(vtk_files)} VTK files")
    
    # Load first file for data ranges
    first_data = load_vtk_file(vtk_files[0])
    
    berry_mag_array = first_data.GetPointData().GetScalars("BerryMagnitude")
    if berry_mag_array is None:
        print("Error: 'BerryMagnitude' field not found")
        return
    berry_mag_range = berry_mag_array.GetRange()
    print(f"Berry Curvature Magnitude range: {berry_mag_range}")
    
    energy_array = first_data.GetPointData().GetScalars("Energy")
    if energy_array is None:
        print("Error: 'Energy' field not found")
        return
    energy_range = energy_array.GetRange()
    print(f"Energy range: {energy_range}")
    
    # Setup renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1920, 1080)
    render_window.SetOffScreenRendering(1)  # Offscreen for video
    
    # Camera setup
    camera = renderer.GetActiveCamera()
    camera.SetPosition(10, 10, 8)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    
    # Video writer setup
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    
    video_filename = "weyl_animation.ogv"
    writer = vtk.vtkOggTheoraWriter()
    writer.SetFileName(video_filename)
    writer.SetRate(24)
    writer.SetQuality(2)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Start()
    
    print(f"\nGenerating video: {video_filename}")
    print(f"Data frames: {len(vtk_files)}")
    
    # Ray for ray-tracing visualization
    ray_origin = np.array([-3.0, -3.0, -3.0])
    ray_direction = np.array([1.0, 1.0, 1.0])
    
    # Loop settings for longer video
    num_loops = 5  # Forward and backward loops
    total_frames = 0
    
    # Create frame sequence: forward, backward, forward, backward...
    frame_sequence = []
    for loop in range(num_loops):
        if loop % 2 == 0:
            frame_sequence.extend(range(len(vtk_files)))  # Forward
        else:
            frame_sequence.extend(range(len(vtk_files) - 1, -1, -1))  # Backward
    
    print(f"Total frames: {len(frame_sequence)} (~{len(frame_sequence)/24:.1f} seconds)")
    
    # Render each frame
    for render_idx, frame_idx in enumerate(frame_sequence):
        vtk_file = vtk_files[frame_idx]
        
        if render_idx % 10 == 0:
            print(f"Rendering frame {render_idx + 1}/{len(frame_sequence)}")
        
        current_data = load_vtk_file(vtk_file)
        
        # Clear previous actors
        renderer.RemoveAllViewProps()
        
        # 1. Direct Volume Rendering (ray casting)
        volume = create_volume_renderer(current_data, berry_mag_range)
        renderer.AddVolume(volume)
        
        # 2. Indirect Volume Rendering (isosurfaces)
        isosurfaces = create_isosurfaces(current_data, berry_mag_range)
        renderer.AddActor(isosurfaces)
        
        # 3. Streamlines (stream functions)
        streamlines = create_streamlines(current_data, num_seeds=150)
        renderer.AddActor(streamlines)
        
        # 4. Cross-section with contours
        slice_actor = create_slice(current_data, [0, 0, 0], [0, 0, 1], show_contours=True)
        renderer.AddActor(slice_actor)
        
        # 5. Ray-tracing visualization
        ray_vis = create_ray_visualization(current_data, ray_origin, ray_direction, berry_mag_range)
        renderer.AddActor(ray_vis)
        
        # 6. Weyl point markers
        weyl_markers = create_weyl_point_markers(current_data)
        if weyl_markers:
            renderer.AddActor(weyl_markers)
        
        # Add frame info text
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Weyl Semimetal - Berry Curvature\n"
                           f"Frame {render_idx + 1}/{len(frame_sequence)}\n"
                           f"M parameter: {1.0 + frame_idx * 0.1:.2f}\n"
                           f"Red: Weyl nodes | Yellow: Ray path\n"
                           f"Streamlines show Berry curvature flow")
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 1)
        text_actor.SetPosition(20, 20)
        renderer.AddActor2D(text_actor)
        
        # Slow camera rotation for animation
        angle = render_idx * 1.5  # degrees per frame
        radius = 12
        camera.SetPosition(
            radius * np.cos(np.radians(angle)),
            radius * np.sin(np.radians(angle)),
            8
        )
        
        renderer.ResetCameraClippingRange()
        render_window.Render()
        
        # Capture frame
        w2i.Modified()
        w2i.Update()
        writer.Write()
    
    writer.End()
    
    print(f"\n{'='*60}")
    print(f"Video saved: {video_filename}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()