"""
Scene building utilities for the PalletTask environment.

Extracts scene setup, pallet mesh spawning, lighting, floor visual, and
mockup-mode physics overrides from the monolithic pallet_task.py.
"""

from __future__ import annotations


# =============================================================================
# Stage / USD Helpers  (Robust, version-safe)
# =============================================================================

def _get_stage():
    """Return the active Omniverse USD stage."""
    import omni.usd
    return omni.usd.get_context().get_stage()


def _is_prim_path_valid(path: str) -> bool:
    """Check whether *path* resolves to a valid USD prim."""
    try:
        stage = _get_stage()
        if not stage:
            return False
        return stage.GetPrimAtPath(path).IsValid()
    except Exception:
        return False


def _create_prim(path: str, prim_type: str, attributes: dict = None):
    """Create (or return existing) a USD prim with optional attributes."""
    try:
        stage = _get_stage()
        if not stage:
            return None
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            prim = stage.DefinePrim(path, prim_type)
        if attributes:
            from pxr import Sdf, Gf
            for k, v in attributes.items():
                if isinstance(v, float):
                    prim.CreateAttribute(k, Sdf.ValueTypeNames.Float).Set(v)
                elif isinstance(v, tuple) and len(v) == 3:
                    prim.CreateAttribute(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        return prim
    except Exception as e:
        print(f"[WARNING] _create_prim failed for {path}: {e}")
        return None


# =============================================================================
# Scene Setup
# =============================================================================

def setup_scene(cfg, scene):
    """
    Configure stage-level scene objects.

    Handles:
      1. Ground plane spawning (with visual material).
      2. Stage lighting for headless rendering.
      3. Visual floor slab.
      4. Container prim creation for Boxes/Pallet.
      5. Optional pallet mesh visual (STL→USD).
      6. Optional mockup-mode physics overrides.

    Args:
        cfg: ``PalletTaskCfg`` instance.
        scene: The ``InteractiveScene`` instance (provides ``env_prim_paths``).
    """
    import omni.usd

    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
    from isaaclab.sim.spawners.shapes import CuboidCfg

    stage = omni.usd.get_context().get_stage()

    def is_prim_path_valid(path: str) -> bool:
        try:
            prim = stage.GetPrimAtPath(path)
            return prim.IsValid()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 1. Ground plane
    # ------------------------------------------------------------------
    _ground_kwargs: dict = {}
    try:
        _ground_kwargs["visual_material"] = PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.53, 0.50),
            roughness=0.92,
            metallic=0.0,
        )
        _ground_cfg = GroundPlaneCfg(**_ground_kwargs)
    except TypeError:
        _ground_kwargs.pop("visual_material", None)
        _has_color = hasattr(GroundPlaneCfg, "color") or "color" in {
            f.name for f in getattr(GroundPlaneCfg, "__dataclass_fields__", {}).values()
        }
        if _has_color:
            _ground_kwargs["color"] = (0.55, 0.53, 0.50)
        _ground_cfg = GroundPlaneCfg(**_ground_kwargs)
        print("[INFO] GroundPlaneCfg: visual_material unsupported, using fallback")

    spawn_ground_plane(
        "/World/groundPlane",
        _ground_cfg,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    # ------------------------------------------------------------------
    # 2. Stage lighting (DomeLight + DistantLight)
    # ------------------------------------------------------------------
    light_path = "/World/DomeLight"
    if not _is_prim_path_valid(light_path):
        _create_prim(
            light_path, "DomeLight",
            attributes={"inputs:intensity": 3000.0, "inputs:color": (1.0, 1.0, 1.0)},
        )
        print("[INFO] Created DomeLight at /World/DomeLight for headless rendering")

    dist_light_path = "/World/DistantLight"
    if not _is_prim_path_valid(dist_light_path):
        _create_prim(
            dist_light_path, "DistantLight",
            attributes={
                "inputs:intensity": 5000.0,
                "inputs:color": (1.0, 0.98, 0.95),
                "inputs:angle": 1.0,
            },
        )
        print("[INFO] Created DistantLight at /World/DistantLight for headless rendering")

    # ------------------------------------------------------------------
    # 3. Visual floor slab (thin prim below z=0)
    # ------------------------------------------------------------------
    floor_path = "/World/FloorVisual"
    if cfg.floor_visual_enabled and not _is_prim_path_valid(floor_path):
        _spawn_floor_visual(cfg, floor_path, CuboidCfg, PreviewSurfaceCfg)

    # ------------------------------------------------------------------
    # 4. Container Xform prims for RigidObjectCollection
    # ------------------------------------------------------------------
    source_env_path = scene.env_prim_paths[0]
    boxes_path = f"{source_env_path}/Boxes"
    if not _is_prim_path_valid(boxes_path):
        _create_prim(boxes_path, "Xform")

    # ------------------------------------------------------------------
    # 5. Optional visual pallet mesh (STL→USD)
    # ------------------------------------------------------------------
    if cfg.use_pallet_mesh_visual:
        spawn_pallet_mesh_visual(cfg, source_env_path)

    # ------------------------------------------------------------------
    # 6. Optional mockup-mode physics overrides
    # ------------------------------------------------------------------
    if cfg.mockup_mode:
        apply_mockup_physics(cfg, source_env_path)


# =============================================================================
# Floor Visual Helper
# =============================================================================

def _spawn_floor_visual(cfg, floor_path, CuboidCfg, PreviewSurfaceCfg):
    """Spawn a thin visual floor slab."""
    _fsx, _fsy = cfg.floor_size_xy
    _ft = cfg.floor_thickness
    _fc = cfg.floor_color
    _fz = -_ft / 2.0 - 0.001

    _floor_spawned = False
    try:
        floor_spawner = CuboidCfg(
            size=(_fsx, _fsy, _ft),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=_fc, roughness=0.9, metallic=0.0,
            ),
        )
        floor_spawner.func(
            floor_path, floor_spawner,
            translation=(0.0, 0.0, _fz),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        _floor_spawned = True
        print(f"[INFO] Spawned visual floor slab via CuboidCfg ({_fsx}x{_fsy}x{_ft}, color={_fc})")
    except Exception as e:
        print(f"[INFO] CuboidCfg visual-only floor failed ({e}), trying USD fallback")

    if not _floor_spawned:
        try:
            from pxr import UsdGeom, UsdShade, Sdf, Gf
            stage = _get_stage()
            cube_prim = _create_prim(floor_path, "Cube")
            cube_prim.GetAttribute("size").Set(1.0)
            xform = UsdGeom.Xformable(cube_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _fz))
            xform.AddScaleOp().Set(Gf.Vec3f(_fsx, _fsy, _ft))
            mat_path = f"{floor_path}/ConcreteMat"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_fc))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(mat)
            print(f"[INFO] Spawned visual floor slab via USD fallback ({_fsx}x{_fsy}x{_ft}, color={_fc})")
        except Exception as e2:
            print(f"[WARNING] Both floor strategies failed: {e2}")


# =============================================================================
# Pallet Mesh Visual (STL→USD)
# =============================================================================

def spawn_pallet_mesh_visual(cfg, source_env_path: str):
    """
    Spawn visual-only pallet mesh from STL file with auto-centering.

    Pipeline:
      1. Resolve STL path (robust glob fallback)
      2. Convert STL→USD via MeshConverter (cached)
      3. Spawn at origin
      4. Compute world bbox via UsdGeom.BBoxCache
      5. Auto-center XY on pallet collider + align Z base to collider top
      6. Apply wood material + hide cuboid pallet visual
    """
    import os
    import glob
    import hashlib

    # --- 1. Resolve STL path ---
    stl_path = cfg.pallet_mesh_stl_path
    if not os.path.isabs(stl_path):
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stl_path = os.path.join(os.path.dirname(pkg_dir), stl_path)

    if not os.path.exists(stl_path):
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(os.path.dirname(pkg_dir), "assets")
        stl_files = sorted(
            glob.glob(os.path.join(assets_dir, "*.stl"))
            + glob.glob(os.path.join(assets_dir, "*.STL"))
        )
        if stl_files:
            pallet_files = [f for f in stl_files if "pallet" in os.path.basename(f).lower()]
            stl_path = pallet_files[0] if pallet_files else stl_files[0]
            print(f"[INFO] Auto-selected pallet STL: {os.path.basename(stl_path)}")
        else:
            print(f"[WARNING] No STL files found in {assets_dir}, skipping pallet mesh")
            return
    print(f"[INFO] Using pallet STL: {stl_path}")

    # --- 2. Convert STL→USD (cached) ---
    cache_dir = cfg.pallet_mesh_cache_dir
    if not os.path.isabs(cache_dir):
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(os.path.dirname(pkg_dir), cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    stl_hash = hashlib.md5(open(stl_path, "rb").read()).hexdigest()[:8]
    scale_str = "_".join(f"{s:.4f}" for s in cfg.pallet_mesh_scale)
    usd_name = f"pallet_mesh_{stl_hash}_{scale_str}.usd"
    usd_path = os.path.join(cache_dir, usd_name)

    if not os.path.exists(usd_path):
        try:
            from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
            converter_cfg = MeshConverterCfg(
                asset_path=stl_path, usd_dir=cache_dir, usd_file_name=usd_name,
                force_usd_conversion=False, make_instanceable=False,
            )
            converter = MeshConverter(converter_cfg)
            usd_path = converter.usd_path
            print(f"[INFO] Converted pallet mesh STL→USD: {usd_path}")
        except Exception as e:
            print(f"[WARNING] Failed to convert pallet mesh: {e}")
            return
    else:
        print(f"[INFO] Using cached pallet mesh USD: {usd_path}")

    # --- 3. Spawn at origin ---
    mesh_prim_path = f"{source_env_path}/PalletMeshVisual"
    try:
        from isaaclab.sim.spawners.from_files import UsdFileCfg
        spawner = UsdFileCfg(
            usd_path=usd_path, scale=cfg.pallet_mesh_scale,
            rigid_props=None, collision_props=None,
        )
        spawner.func(
            mesh_prim_path, spawner,
            translation=(0.0, 0.0, 0.0),
            orientation=cfg.pallet_mesh_offset_quat_wxyz,
        )
        print(f"[INFO] Spawned visual pallet mesh at {mesh_prim_path}")
    except Exception as e:
        print(f"[WARNING] Failed to spawn pallet mesh visual: {e}")
        return

    # --- 4 & 5. Auto-centering ---
    pallet_prim_path = f"{source_env_path}/Pallet"
    dx, dy, dz = 0.0, 0.0, 0.0
    try:
        from pxr import UsdGeom, Usd, Gf
        stage = _get_stage()

        pallet_prim = stage.GetPrimAtPath(pallet_prim_path)
        collider_center = Gf.Vec3d(0.0, 0.0, 0.075)
        pallet_half_h = 0.075
        if pallet_prim.IsValid():
            pallet_xform = UsdGeom.Xformable(pallet_prim)
            local_mat = pallet_xform.GetLocalTransformation()
            collider_center = local_mat.ExtractTranslation()
            pallet_half_h = 0.15 / 2.0
            print(f"[INFO] Pallet collider center from USD: "
                  f"({collider_center[0]:.4f}, {collider_center[1]:.4f}, {collider_center[2]:.4f})")
        else:
            print("[WARNING] Pallet prim not found, using fallback center (0,0,0.075)")

        collider_bottom_z = collider_center[2] - pallet_half_h

        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if mesh_prim.IsValid():
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=False,
            )
            bbox = cache.ComputeWorldBound(mesh_prim)
            aligned = bbox.ComputeAlignedRange()
            min_pt = aligned.GetMin()
            max_pt = aligned.GetMax()
            center = (min_pt + max_pt) / 2.0

            print(f"[INFO] Pallet mesh bbox:")
            print(f"  min = ({min_pt[0]:.4f}, {min_pt[1]:.4f}, {min_pt[2]:.4f})")
            print(f"  max = ({max_pt[0]:.4f}, {max_pt[1]:.4f}, {max_pt[2]:.4f})")
            print(f"  center = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")

            if cfg.pallet_mesh_auto_center:
                dx = collider_center[0] - center[0]
                dy = collider_center[1] - center[1]
            if cfg.pallet_mesh_auto_align_z:
                dz = collider_bottom_z - min_pt[2]

            print(f"[INFO] Auto-correction: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

            corrected_min_z = min_pt[2] + dz
            if corrected_min_z > 0.005:
                extra_dz = -corrected_min_z
                dz += extra_dz
                print(f"[WARNING] Mesh still floating (min_z={corrected_min_z:.4f}), "
                      f"applying extra dz={extra_dz:.4f}")
        else:
            print("[WARNING] Pallet mesh prim not valid for bbox computation")
    except Exception as e:
        print(f"[WARNING] BBox computation failed, using manual offset: {e}")

    # --- 5c. Apply correction + user offset ---
    user_off = cfg.pallet_mesh_offset_pos
    final_x = dx + user_off[0]
    final_y = dy + user_off[1]
    final_z = dz + user_off[2]

    try:
        from pxr import UsdGeom, Gf
        stage = _get_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if mesh_prim.IsValid():
            xformable = UsdGeom.Xformable(mesh_prim)
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(final_x, final_y, final_z))
            print(f"[INFO] Pallet mesh final position: ({final_x:.4f}, {final_y:.4f}, {final_z:.4f})")
    except Exception as e:
        print(f"[WARNING] Could not reposition pallet mesh: {e}")

    # --- 6a. Apply wood material ---
    try:
        from pxr import UsdShade, Sdf, Gf
        stage = _get_stage()
        mat_path = f"{mesh_prim_path}/WoodMaterial"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.72, 0.55, 0.35)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if mesh_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)
        print("[INFO] Applied wood material to pallet mesh")
    except Exception as e:
        print(f"[WARNING] Could not apply wood material: {e}")

    # --- 6b. Hide cuboid pallet visual ---
    pallet_visual_path = f"{source_env_path}/Pallet"
    try:
        from pxr import UsdGeom
        stage = _get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_visual_path)
        if pallet_prim.IsValid():
            imageable = UsdGeom.Imageable(pallet_prim)
            imageable.MakeInvisible()
            print(f"[INFO] Hidden cuboid pallet visual at {pallet_visual_path}")
    except Exception as e:
        print(f"[WARNING] Could not hide cuboid pallet: {e}")


# =============================================================================
# Mockup-Mode Physics Overrides
# =============================================================================

def apply_mockup_physics(cfg, source_env_path: str):
    """
    Apply mockup-mode physics overrides to box prims via USD attributes.

    Sets high friction, zero restitution, linear/angular damping,
    solver iterations, and velocity clamping on all box prims.
    Only called when ``mockup_mode=True``; does not affect training defaults.
    """
    try:
        from pxr import UsdPhysics, PhysxSchema, Sdf
        stage = _get_stage()
        if not stage:
            return

        # --- Pallet collider verification ---
        pallet_path = f"{source_env_path}/Pallet"
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        if pallet_prim.IsValid():
            has_collision = pallet_prim.HasAPI(UsdPhysics.CollisionAPI)
            has_rb = pallet_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            is_kinematic = False
            if has_rb:
                rb = UsdPhysics.RigidBodyAPI(pallet_prim)
                kin_attr = rb.GetKinematicEnabledAttr()
                is_kinematic = kin_attr.Get() if kin_attr else False
            from pxr import UsdGeom, Gf
            xformable = UsdGeom.Xformable(pallet_prim)
            local_mat = xformable.GetLocalTransformation()
            center = local_mat.ExtractTranslation()
            pallet_bottom_z = center[2] - 0.075
            print(f"[INFO] Pallet collider verification:")
            print(f"  path={pallet_path}")
            print(f"  CollisionAPI={has_collision}, RigidBodyAPI={has_rb}, kinematic={is_kinematic}")
            print(f"  center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), bottom_z={pallet_bottom_z:.4f}")
        else:
            print(f"[WARNING] Pallet prim not found at {pallet_path}")

        applied_count = 0
        for i in range(cfg.max_boxes):
            box_path = f"{source_env_path}/Boxes/box_{i}"
            box_prim = stage.GetPrimAtPath(box_path)
            if not box_prim.IsValid():
                continue

            # Physics material (friction + restitution)
            mat_path = f"{box_path}/MockupPhysMat"
            UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(Sdf.Path(mat_path)))
            mat_prim = stage.GetPrimAtPath(mat_path)
            mat_api = UsdPhysics.MaterialAPI(mat_prim)
            mat_api.CreateStaticFrictionAttr().Set(cfg.mockup_box_static_friction)
            mat_api.CreateDynamicFrictionAttr().Set(cfg.mockup_box_dynamic_friction)
            mat_api.CreateRestitutionAttr().Set(cfg.mockup_box_restitution)

            UsdPhysics.MaterialAPI.Apply(box_prim)
            phys_mat = UsdPhysics.MaterialAPI(box_prim)
            phys_mat.CreateStaticFrictionAttr().Set(cfg.mockup_box_static_friction)
            phys_mat.CreateDynamicFrictionAttr().Set(cfg.mockup_box_dynamic_friction)
            phys_mat.CreateRestitutionAttr().Set(cfg.mockup_box_restitution)

            # Rigid body stability
            if box_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb_api = PhysxSchema.PhysxRigidBodyAPI(box_prim)
            else:
                rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(box_prim)

            if rb_api is not None:
                rb_api.CreateLinearDampingAttr().Set(cfg.mockup_box_linear_damping)
                rb_api.CreateAngularDampingAttr().Set(cfg.mockup_box_angular_damping)
                rb_api.CreateMaxLinearVelocityAttr().Set(cfg.mockup_box_max_linear_velocity)
                rb_api.CreateMaxAngularVelocityAttr().Set(cfg.mockup_box_max_angular_velocity)
                rb_api.CreateSolverPositionIterationCountAttr().Set(cfg.mockup_solver_position_iterations)
                rb_api.CreateSolverVelocityIterationCountAttr().Set(cfg.mockup_solver_velocity_iterations)
                rb_api.CreateMaxDepenetrationVelocityAttr().Set(cfg.mockup_max_depenetration_velocity)
                if cfg.mockup_enable_ccd:
                    rb_api.CreateEnableCCDAttr().Set(True)

            # Collision properties
            if box_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                col_api = PhysxSchema.PhysxCollisionAPI(box_prim)
            else:
                col_api = PhysxSchema.PhysxCollisionAPI.Apply(box_prim)
            if col_api is not None:
                col_api.CreateContactOffsetAttr().Set(cfg.mockup_contact_offset)
                col_api.CreateRestOffsetAttr().Set(cfg.mockup_rest_offset)

            applied_count += 1

        # Pallet surface friction
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        if pallet_prim.IsValid():
            UsdPhysics.MaterialAPI.Apply(pallet_prim)
            pal_mat = UsdPhysics.MaterialAPI(pallet_prim)
            pal_mat.CreateStaticFrictionAttr().Set(cfg.mockup_pallet_static_friction)
            pal_mat.CreateDynamicFrictionAttr().Set(cfg.mockup_pallet_dynamic_friction)
            pal_mat.CreateRestitutionAttr().Set(0.0)
            print(f"[INFO] Applied mockup friction to pallet collider: "
                  f"static={cfg.mockup_pallet_static_friction}, "
                  f"dynamic={cfg.mockup_pallet_dynamic_friction}")

        print(f"[INFO] Applied mockup physics to {applied_count} box prims")
        print(f"  friction: static={cfg.mockup_box_static_friction}, "
              f"dynamic={cfg.mockup_box_dynamic_friction}, "
              f"restitution={cfg.mockup_box_restitution}")
        print(f"  damping: linear={cfg.mockup_box_linear_damping}, "
              f"angular={cfg.mockup_box_angular_damping}")
        print(f"  velocity clamp: linear={cfg.mockup_box_max_linear_velocity} m/s, "
              f"angular={cfg.mockup_box_max_angular_velocity} rad/s")
        print(f"  depenetration: max_vel={cfg.mockup_max_depenetration_velocity} m/s, "
              f"CCD={cfg.mockup_enable_ccd}")
        print(f"  solver iterations: pos={cfg.mockup_solver_position_iterations}, "
              f"vel={cfg.mockup_solver_velocity_iterations}")

    except ImportError as e:
        print(f"[WARNING] PhysX/USD schemas not available for mockup physics: {e}")
    except Exception as e:
        print(f"[WARNING] Failed to apply mockup physics: {e}")
        import traceback
        traceback.print_exc()
