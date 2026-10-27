import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_RENDERER = None


# ---------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------

def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)

    n = np.linalg.norm(q)
    if n > 0:
        q = q / n

    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    return R


def pose7_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    T[:3, 3] = pose[:3]

    return T


def pos_quat_to_matrix(pos, quat):
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(quat)
    T[:3, 3] = pos

    return T


def triangulate(counts, indices):
    faces = []
    k = 0

    for n in counts:
        n = int(n)
        face = indices[k:k+n]
        k += n

        for i in range(1, n - 1):
            faces.append([
                int(face[0]),
                int(face[i]),
                int(face[i+1]),
            ])

    return np.asarray(faces, dtype=np.int64)


# ---------------------------------------------------------
# Runtime object discovery
# ---------------------------------------------------------

def find_articulations(robot_manager):
    found = []

    def inspect(v):
        try:
            if (
                hasattr(v, "data")
                and hasattr(v.data, "body_link_pose_w")
                and hasattr(v.data, "body_names")
            ):
                if all(id(v) != id(x) for x in found):
                    found.append(v)
                return
        except Exception:
            pass

        if isinstance(v, dict):
            for x in v.values():
                inspect(x)

        elif isinstance(v, (list, tuple)):
            for x in v:
                inspect(x)

    for value in robot_manager.__dict__.values():
        inspect(value)

    return found


def find_runtime_scene_objects(scene_manager):
    """
    Find RoboDojo scene objects which:
      1) own a USD prim path
      2) provide runtime get_world_pose()

    Works without assuming the exact SceneManager container name.
    """
    found = []
    seen = set()

    def inspect(v):
        if v is None:
            return

        oid = id(v)

        if oid in seen:
            return

        if hasattr(v, "get_world_pose"):
            prim_path = getattr(v, "usd_prim_path", None)

            if prim_path is None:
                prim_path = getattr(v, "_prim_path", None)

            if prim_path is None:
                prim_path = getattr(v, "prim_path", None)

            if prim_path is not None:
                seen.add(oid)
                found.append((str(prim_path), v))
                return

        if isinstance(v, dict):
            for x in v.values():
                inspect(x)

        elif isinstance(v, (list, tuple)):
            for x in v:
                inspect(x)

    # Only inspect likely object collections to avoid crawling the whole manager.
    for name, value in scene_manager.__dict__.items():
        if (
            "object" in name.lower()
            or "rigid" in name.lower()
            or "dynamic" in name.lower()
            or "geometry" in name.lower()
            or "articulation" in name.lower()
        ):
            inspect(value)

    # Longest path first for unambiguous prefix matching.
    found.sort(key=lambda x: len(x[0]), reverse=True)

    return found


# ---------------------------------------------------------
# Renderer
# ---------------------------------------------------------

class RuntimeEGLRenderer:

    def __init__(self, robot_manager, scene_manager,
                 env_root="/World/envs/env_0"):

        import omni.usd
        import trimesh
        import pyrender
        from pxr import UsdGeom

        self.robot_manager = robot_manager
        self.scene_manager = scene_manager

        self.UsdGeom = UsdGeom
        self.pyrender = pyrender

        self.stage = omni.usd.get_context().get_stage()

        self.scene = pyrender.Scene(
            bg_color=[35, 35, 35, 255],
            ambient_light=[0.38, 0.38, 0.38],
        )

        # ---------------------------------------------
        # Isaac Lab articulations
        # ---------------------------------------------

        self.articulations = find_articulations(robot_manager)

        print(
            f"[EGL3] articulations={len(self.articulations)}"
        )

        if len(self.articulations) < 2:
            print(
                "[EGL3 WARN] expected two X5 articulations"
            )

        self.body_lookup = {}

        for art_idx, art in enumerate(self.articulations):

            names = list(art.data.body_names)

            print(
                f"[EGL3] robot{art_idx} bodies={names}"
            )

            for body_idx, body_name in enumerate(names):
                self.body_lookup.setdefault(
                    str(body_name), []
                ).append(
                    (art, body_idx)
                )

        # ---------------------------------------------
        # RoboDojo dynamic scene objects
        # ---------------------------------------------

        self.runtime_objects = find_runtime_scene_objects(
            scene_manager
        )

        print(
            f"[EGL3] runtime scene objects="
            f"{len(self.runtime_objects)}"
        )

        for path, obj in self.runtime_objects[:20]:
            print(
                "[EGL3 OBJECT]",
                type(obj).__name__,
                path
            )

        # ---------------------------------------------
        # Mesh import
        # ---------------------------------------------

        self.robot_nodes = []
        self.object_nodes = []
        self.static_nodes = []

        xform_cache = UsdGeom.XformCache()

        imported = 0
        skipped_collision = 0

        for prim in self.stage.Traverse():

            path = str(prim.GetPath())

            if not path.startswith(env_root):
                continue

            if not prim.IsA(UsdGeom.Mesh):
                continue

            # Critical performance optimization:
            # never render collision geometry.
            if "/collisions/" in path:
                skipped_collision += 1
                continue

            mesh = UsdGeom.Mesh(prim)

            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not points or not counts or not indices:
                continue

            vertices = np.asarray(
                points,
                dtype=np.float64
            )

            faces = triangulate(
                counts,
                indices
            )

            if (
                len(vertices) == 0
                or len(faces) == 0
            ):
                continue

            imported += 1

            # Stable pseudo material.
            h = abs(hash(path))

            rgba = np.array([
                90 + h % 130,
                90 + (h // 7) % 130,
                90 + (h // 49) % 130,
                255,
            ], dtype=np.float32) / 255.0

            tm = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )

            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=rgba,
                metallicFactor=0.0,
                roughnessFactor=0.82,
            )

            pm = pyrender.Mesh.from_trimesh(
                tm,
                material=material,
                smooth=False,
            )

            usd_world_row = np.asarray(
                xform_cache.GetLocalToWorldTransform(
                    prim
                ),
                dtype=np.float64,
            )

            # USD/Gf is row-vector convention here.
            usd_world = usd_world_row.T

            # -----------------------------------------
            # A. Robot link?
            # -----------------------------------------

            ancestor = prim
            body_match = None

            while ancestor and ancestor.IsValid():

                name = str(
                    ancestor.GetName()
                )

                if name in self.body_lookup:
                    body_match = name
                    break

                parent = ancestor.GetParent()

                if (
                    not parent
                    or parent == ancestor
                ):
                    break

                ancestor = parent

            if body_match is not None:

                candidates = self.body_lookup[
                    body_match
                ]

                best = None
                best_err = 1e99

                for art, body_idx in candidates:

                    pose = (
                        art.data.body_link_pose_w[
                            0, body_idx
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    body_world = pose7_to_matrix(
                        pose
                    )

                    err = np.linalg.norm(
                        body_world[:3, 3]
                        - usd_world[:3, 3]
                    )

                    if err < best_err:
                        best_err = err
                        best = (
                            art,
                            body_idx,
                            body_world,
                        )

                art, body_idx, body_world = best

                local_tf = (
                    np.linalg.inv(body_world)
                    @ usd_world
                )

                node = self.scene.add(
                    pm,
                    pose=usd_world
                )

                self.robot_nodes.append(
                    (
                        art,
                        body_idx,
                        local_tf,
                        node,
                        path,
                    )
                )

                continue

            # -----------------------------------------
            # B. Runtime scene object?
            # -----------------------------------------

            obj_match = None

            for object_root, obj in self.runtime_objects:

                if (
                    path == object_root
                    or path.startswith(
                        object_root.rstrip("/")
                        + "/"
                    )
                ):
                    obj_match = (
                        object_root,
                        obj
                    )
                    break

            if obj_match is not None:

                object_root, obj = obj_match

                try:
                    pos, quat = obj.get_world_pose()

                    if hasattr(pos, "detach"):
                        pos = (
                            pos.detach()
                            .cpu()
                            .numpy()
                        )

                    if hasattr(quat, "detach"):
                        quat = (
                            quat.detach()
                            .cpu()
                            .numpy()
                        )

                    object_world = (
                        pos_quat_to_matrix(
                            pos,
                            quat
                        )
                    )

                    local_tf = (
                        np.linalg.inv(
                            object_world
                        )
                        @ usd_world
                    )

                    node = self.scene.add(
                        pm,
                        pose=usd_world
                    )

                    self.object_nodes.append(
                        (
                            obj,
                            local_tf,
                            node,
                            path,
                        )
                    )

                    continue

                except Exception as e:
                    print(
                        "[EGL3 WARN] object runtime "
                        "pose failed:",
                        path,
                        repr(e)
                    )

            # -----------------------------------------
            # C. Truly static geometry
            # -----------------------------------------

            node = self.scene.add(
                pm,
                pose=usd_world
            )

            self.static_nodes.append(
                node
            )

        print(
            "[EGL3] "
            f"visual_meshes={imported}, "
            f"collision_skipped={skipped_collision}, "
            f"robot_dynamic={len(self.robot_nodes)}, "
            f"object_dynamic={len(self.object_nodes)}, "
            f"static={len(self.static_nodes)}"
        )

        # ---------------------------------------------
        # Camera intrinsics
        # ---------------------------------------------

        self.width = 224
        self.height = 224

        # Gemini 345Lg scaled from 640x480.
        fx_native = (
            640.0 * 10.0 / 22.212
        )

        fy_native = (
            480.0 * 10.0 / 14.266
        )

        fx = (
            fx_native
            * self.width
            / 640.0
        )

        fy = (
            fy_native
            * self.height
            / 480.0
        )

        self.camera = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=self.width / 2.0,
            cy=self.height / 2.0,
            znear=0.005,
            zfar=10.0,
        )

        # Exact head camera.
        theta = np.deg2rad(30.0)

        self.head_pose = np.eye(
            4,
            dtype=np.float64
        )

        self.head_pose[:3, :3] = np.array([
            [1.0, 0.0,            0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta),  np.cos(theta)],
        ])

        self.head_pose[:3, 3] = np.array([
            0.0,
            -0.41,
            1.308
        ])

        self.camera_node = self.scene.add(
            self.camera,
            pose=self.head_pose
        )

        light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=3.0
        )

        self.light_node = self.scene.add(
            light,
            pose=self.head_pose
        )

        self.renderer = (
            pyrender.OffscreenRenderer(
                self.width,
                self.height
            )
        )

        # ---------------------------------------------
        # Identify wrist camera bodies
        # ---------------------------------------------

        self.wrist_bodies = []

        for art in self.articulations:

            names = list(
                art.data.body_names
            )

            if "camera" not in names:
                self.wrist_bodies.append(
                    None
                )
                continue

            body_idx = names.index(
                "camera"
            )

            self.wrist_bodies.append(
                (art, body_idx)
            )

        # Sort robots left -> right using base_link X.
        sortable = []

        for wb in self.wrist_bodies:

            if wb is None:
                continue

            art, camera_idx = wb

            names = list(
                art.data.body_names
            )

            base_idx = (
                names.index("base_link")
                if "base_link" in names
                else 0
            )

            base_pose = (
                art.data.body_link_pose_w[
                    0, base_idx
                ]
                .detach()
                .cpu()
                .numpy()
            )

            sortable.append(
                (
                    float(base_pose[0]),
                    art,
                    camera_idx,
                )
            )

        sortable.sort(
            key=lambda x: x[0]
        )

        self.left_wrist = (
            (sortable[0][1], sortable[0][2])
            if len(sortable) >= 1
            else None
        )

        self.right_wrist = (
            (sortable[-1][1], sortable[-1][2])
            if len(sortable) >= 2
            else None
        )

        print(
            "[EGL3] wrist cameras:",
            "left=",
            self.left_wrist is not None,
            "right=",
            self.right_wrist is not None,
        )

        # Exact X5 wrist-camera local pose from:
        # Assets/Robots/x5/robot_config.yml
        #
        # link: camera
        # pos : [0, 0, 0.001]
        # ori : [0, -80, -90] deg
        #
        # Use RoboDojo's own Euler -> quaternion convention
        # instead of guessing rotation order.
        import torch
        from utils.rotations import euler_angles_to_quat

        wrist_q = euler_angles_to_quat(
            torch.tensor(
                [0.0, -80.0, -90.0],
                dtype=torch.float32
            ),
            degrees=True,
        )

        if hasattr(wrist_q, "detach"):
            wrist_q = (
                wrist_q.detach()
                .cpu()
                .numpy()
            )

        self.wrist_local_pose = pos_quat_to_matrix(
            np.array(
                [0.0, 0.0, 0.001],
                dtype=np.float64
            ),
            wrist_q
        )

        print(
            "[EGL3 WRIST LOCAL] "
            "link=camera, "
            "pos=[0,0,0.001], "
            "ori=[0,-80,-90]"
        )

        # ---------------------------------------------
        # Rate limiting
        # ---------------------------------------------

        self.obs_counter = 0
        self.render_counter = 0
        self.render_every = 10

        self.cached = None

        print(
            "[EGL3 READY] "
            "224x224, "
            "render_every=10"
        )

    # -------------------------------------------------
    # Runtime scene update
    # -------------------------------------------------

    def update_runtime_geometry(self):

        for (
            art,
            body_idx,
            local_tf,
            node,
            path,
        ) in self.robot_nodes:

            pose = (
                art.data.body_link_pose_w[
                    0, body_idx
                ]
                .detach()
                .cpu()
                .numpy()
            )

            body_world = pose7_to_matrix(
                pose
            )

            self.scene.set_pose(
                node,
                pose=(
                    body_world
                    @ local_tf
                )
            )

        for (
            obj,
            local_tf,
            node,
            path,
        ) in self.object_nodes:

            try:
                pos, quat = (
                    obj.get_world_pose()
                )

                if hasattr(pos, "detach"):
                    pos = (
                        pos.detach()
                        .cpu()
                        .numpy()
                    )

                if hasattr(quat, "detach"):
                    quat = (
                        quat.detach()
                        .cpu()
                        .numpy()
                    )

                object_world = (
                    pos_quat_to_matrix(
                        pos,
                        quat
                    )
                )

                self.scene.set_pose(
                    node,
                    pose=(
                        object_world
                        @ local_tf
                    )
                )

            except Exception:
                pass

    # -------------------------------------------------
    # Camera poses
    # -------------------------------------------------

    def wrist_pose(self, wrist):

        if wrist is None:
            return None

        art, body_idx = wrist

        pose = (
            art.data.body_link_pose_w[
                0, body_idx
            ]
            .detach()
            .cpu()
            .numpy()
        )

        link_world = pose7_to_matrix(
            pose
        )

        # RoboDojo CameraManager creates the wrist camera as:
        #
        # robotX/camera/Camera_i
        #
        # Therefore camera world pose is:
        #
        # camera_link_world @ camera_local_pose
        #
        return (
            link_world
            @ self.wrist_local_pose
        )

    # -------------------------------------------------
    # Render one camera
    # -------------------------------------------------

    def render_camera(self, pose):

        self.scene.set_pose(
            self.camera_node,
            pose=pose
        )

        self.scene.set_pose(
            self.light_node,
            pose=pose
        )

        rgb, depth = (
            self.renderer.render(
                self.scene
            )
        )

        return rgb, depth

    # -------------------------------------------------
    # Three-camera output
    # -------------------------------------------------

    def render_all(self):

        self.obs_counter += 1

        if (
            self.cached is not None
            and
            (self.obs_counter - 1)
            % self.render_every
            != 0
        ):
            return self.cached

        self.update_runtime_geometry()

        result = {}

        # ---------------- cam_head ----------------

        head_rgb, head_depth = (
            self.render_camera(
                self.head_pose
            )
        )

        result["cam_head"] = head_rgb

        # -------------- left wrist --------------

        left_pose = self.wrist_pose(
            self.left_wrist
        )

        if left_pose is not None:

            left_rgb, left_depth = (
                self.render_camera(
                    left_pose
                )
            )

            result[
                "cam_left_wrist"
            ] = left_rgb

        else:

            left_rgb = np.zeros(
                (
                    self.height,
                    self.width,
                    3
                ),
                dtype=np.uint8
            )

            left_depth = np.zeros(
                (
                    self.height,
                    self.width
                ),
                dtype=np.float32
            )

            result[
                "cam_left_wrist"
            ] = left_rgb

        # -------------- right wrist -------------

        right_pose = self.wrist_pose(
            self.right_wrist
        )

        if right_pose is not None:

            right_rgb, right_depth = (
                self.render_camera(
                    right_pose
                )
            )

            result[
                "cam_right_wrist"
            ] = right_rgb

        else:

            right_rgb = np.zeros(
                (
                    self.height,
                    self.width,
                    3
                ),
                dtype=np.uint8
            )

            right_depth = np.zeros(
                (
                    self.height,
                    self.width
                ),
                dtype=np.float32
            )

            result[
                "cam_right_wrist"
            ] = right_rgb

        self.render_counter += 1

        print(
            "[EGL3 FRAME] "
            f"render={self.render_counter}, "
            f"obs={self.obs_counter}, "
            f"head_std={head_rgb.std():.2f}, "
            f"left_std={left_rgb.std():.2f}, "
            f"right_std={right_rgb.std():.2f}, "
            f"head_depth={(head_depth > 0).sum()}, "
            f"left_depth={(left_depth > 0).sum()}, "
            f"right_depth={(right_depth > 0).sum()}"
        )

        # Save useful inspection frames.
        if self.render_counter in (
            1, 5, 15
        ):

            from PIL import Image

            for name, rgb in result.items():

                path = os.path.expanduser(
                    f"~/egl3_"
                    f"{name}_"
                    f"{self.render_counter:03d}.png"
                )

                Image.fromarray(
                    rgb
                ).save(
                    path
                )

        self.cached = {
            k: v.copy()
            for k, v in result.items()
        }

        return self.cached


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def render_all_cameras(
    robot_manager,
    scene_manager
):
    global _RENDERER

    if _RENDERER is None:
        _RENDERER = RuntimeEGLRenderer(
            robot_manager,
            scene_manager
        )

    return _RENDERER.render_all()
