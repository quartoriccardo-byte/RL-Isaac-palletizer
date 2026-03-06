"""
Shared USD / PhysX helpers for the PalletTask ecosystem.

These functions operate on the USD stage via ``pxr`` APIs and are used by
both ``scene_builder.py`` and ``mockup_video_physics.py``.

All functions are standalone — no class state, no heavy imports at module
level (``pxr`` is imported lazily on first call via ``from pxr import …``).
"""

from __future__ import annotations


# =============================================================================
# Rigid Body Helpers
# =============================================================================

def set_kinematic(stage, prim_path: str, kinematic: bool) -> None:
    """
    Toggle ``kinematic_enabled`` on a rigid body via ``UsdPhysics.RigidBodyAPI``.

    Applies ``RigidBodyAPI`` if missing.  Uses ``Create`` (not ``Get``) to
    guarantee the attribute always exists.
    """
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
    rb.CreateKinematicEnabledAttr().Set(bool(kinematic))


def set_disable_gravity(stage, prim_path: str, disable: bool) -> None:
    """Toggle ``disable_gravity`` on a rigid body via ``PhysxSchema``."""
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    attr = api.GetDisableGravityAttr()
    if not attr or not attr.IsValid():
        attr = api.CreateDisableGravityAttr()
    attr.Set(bool(disable))


def tune_rigid_body(
    stage,
    prim_path: str,
    *,
    lin_damp: float = 2.0,
    ang_damp: float = 2.0,
    max_depen_vel: float = 0.5,
    max_lin_vel: float = 2.0,
    pos_iters: int = 16,
    vel_iters: int = 4,
) -> None:
    """Set PhysX rigid body tuning parameters for stable contacts."""
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    for getter, creator, val in [
        ("GetLinearDampingAttr",                 "CreateLinearDampingAttr",                 float(lin_damp)),
        ("GetAngularDampingAttr",                "CreateAngularDampingAttr",                float(ang_damp)),
        ("GetMaxDepenetrationVelocityAttr",      "CreateMaxDepenetrationVelocityAttr",      float(max_depen_vel)),
        ("GetMaxLinearVelocityAttr",             "CreateMaxLinearVelocityAttr",             float(max_lin_vel)),
        ("GetSolverPositionIterationCountAttr",  "CreateSolverPositionIterationCountAttr",  int(pos_iters)),
        ("GetSolverVelocityIterationCountAttr",  "CreateSolverVelocityIterationCountAttr",  int(vel_iters)),
    ]:
        attr = getattr(api, getter)()
        if not attr or not attr.IsValid():
            attr = getattr(api, creator)()
        attr.Set(val)


# =============================================================================
# Physics Material
# =============================================================================

def set_physics_material(
    stage,
    prim_path: str,
    *,
    static_friction: float = 1.2,
    dynamic_friction: float = 0.9,
    restitution: float = 0.0,
    restitution_combine_mode: str = "min",
) -> None:
    """Apply/update a ``UsdPhysics.MaterialAPI`` and ``PhysxMaterialAPI`` on a prim."""
    from pxr import UsdPhysics, PhysxSchema

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    mat_api = UsdPhysics.MaterialAPI.Apply(prim)
    mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(restitution))

    api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    attr = api.GetRestitutionCombineModeAttr()
    if not attr or not attr.IsValid():
        attr = api.CreateRestitutionCombineModeAttr()
    attr.Set(restitution_combine_mode)


# =============================================================================
# Render Mesh Discovery
# =============================================================================

def get_render_mesh_prim(stage, box_root_prim):
    """
    Return the first renderable ``UsdGeom.Gprim`` under a box root Xform.

    Isaac Lab spawns shape assets with a hierarchy::

        {prim_path}                  — Xform root (rigid body APIs)
        {prim_path}/geometry/mesh    — actual UsdGeom Gprim (renderable)

    Strategy:
      1. Try the well-known child path: ``{root}/geometry/mesh``
      2. Fallback: DFS over all descendants, return first valid Gprim.
      3. Return ``None`` if nothing found.
    """
    from pxr import UsdGeom

    root_path = box_root_prim.GetPath().pathString

    mesh_path = f"{root_path}/geometry/mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if mesh_prim and mesh_prim.IsValid():
        gprim = UsdGeom.Gprim(mesh_prim)
        if gprim:
            return mesh_prim

    for desc in box_root_prim.GetAllDescendants():
        if desc.IsValid() and UsdGeom.Gprim(desc):
            return desc

    return None


# =============================================================================
# Visibility / Purpose Debug Fixer
# =============================================================================

def debug_box_sync_prims(stage, prim_path: str) -> None:
    """
    Inspect and fix visibility/purpose on a box prim and its children.

    If an Imageable prim has ``purpose != 'default'`` or
    ``visibility == 'invisible'``, it is force-fixed so the box renders.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [BOX_SYNC_DBG] prim {prim_path} is INVALID")
        return

    for child_prim in [prim] + list(prim.GetAllChildren()):
        p = child_prim.GetPath().pathString
        img = UsdGeom.Imageable(child_prim)
        if not img:
            continue
        purpose_attr = img.GetPurposeAttr()
        purpose = purpose_attr.Get() if purpose_attr else None
        vis_attr = img.GetVisibilityAttr()
        vis = vis_attr.Get() if vis_attr else None
        print(f"  [BOX_SYNC_DBG]   {p}  purpose={purpose}  visibility={vis}")

        needs_fix = False
        if purpose is not None and purpose != UsdGeom.Tokens.default_:
            print(f"  [BOX_SYNC_DBG]     → fixing purpose to 'default'")
            purpose_attr.Set(UsdGeom.Tokens.default_)
            needs_fix = True
        if vis is not None and vis == UsdGeom.Tokens.invisible:
            print(f"  [BOX_SYNC_DBG]     → calling MakeVisible()")
            img.MakeVisible()
            needs_fix = True
        if needs_fix:
            print(f"  [BOX_SYNC_DBG]     FIXED: {p}")


# =============================================================================
# AABB Geometry Helpers
# =============================================================================

def aabb_overlap(pos_a, dims_a, pos_b, dims_b, margin: float = 0.005) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap (with a small margin).

    Boxes are assumed to be axis-aligned (yaw 0 or 90° → already rotated
    into dims).  The margin prevents false-positive triggers from
    touching-but-not-penetrating surfaces.

    Returns ``True`` if boxes overlap beyond margin.
    """
    for axis in range(3):
        half_a = 0.5 * dims_a[axis] - margin
        half_b = 0.5 * dims_b[axis] - margin
        if abs(pos_a[axis] - pos_b[axis]) >= half_a + half_b:
            return False
    return True


def aabb_intersection_area(pos_a, dims_a, pos_b, dims_b) -> float:
    """Compute the 2D intersection area (XY) of two AABBs."""
    x_overlap = max(
        0.0,
        min(pos_a[0] + dims_a[0] / 2, pos_b[0] + dims_b[0] / 2)
        - max(pos_a[0] - dims_a[0] / 2, pos_b[0] - dims_b[0] / 2),
    )
    y_overlap = max(
        0.0,
        min(pos_a[1] + dims_a[1] / 2, pos_b[1] + dims_b[1] / 2)
        - max(pos_a[1] - dims_a[1] / 2, pos_b[1] - dims_b[1] / 2),
    )
    return float(x_overlap * y_overlap)
