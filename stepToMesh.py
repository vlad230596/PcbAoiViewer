"""STEP assembly -> triangle meshes, one per leaf instance, in board mm (the STEP's own frame).

Self-contained, no project imports: runs in the CAD venv (OpenCascade bindings `cadquery-ocp`), kept out of the
project venv like torch:
    research/cad/venv/Scripts/python.exe stepToMesh.py boards/<board> [--deflection 0.02]
Reads boards/<board>/export/ExportSTEP/*.step, writes boards/<board>/model/
    mesh.npz    vertices float32 (N,3) mm, normals float32 (N,3), triangles uint32 (M,3)
    parts.json  [{name, path, color, vertexStart, vertexCount, triangleStart, triangleCount, bboxMin, bboxMax}]
`path` is the chain of instance/product names from the root, so a component's designator can be found in it.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool


def labelName(label):
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ''


class StepMesher:
    def __init__(self, path, deflection):
        self.deflection = deflection
        self.doc = doc = TDocStd_Document(TCollection_ExtendedString('MDTV-XCAF'))
        XCAFApp_Application.GetApplication_s().InitDocument(doc)
        self.reader = reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        if reader.ReadFile(path) != IFSelect_RetDone:
            raise SystemExit(f'cannot read {path}')
        if not reader.Transfer(doc):
            raise SystemExit(f'cannot transfer {path}')
        self.shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        self.colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
        self.parts, self.verts, self.norms, self.tris = [], [], [], []
        self.nVerts = self.nTris = 0

    def color(self, label, shape):
        c = Quantity_Color()
        for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
            if self.colors.GetColor_s(label, kind, c) or self.colors.GetColor(shape, kind, c):
                return [round(c.Red(), 3), round(c.Green(), 3), round(c.Blue(), 3)]
        return None

    def run(self):
        roots = TDF_LabelSequence()
        self.shapes.GetFreeShapes(roots)
        for i in range(1, roots.Length() + 1):
            self.walk(roots.Value(i), TopLoc_Location(), [], None)

    def walk(self, label, loc, path, inherited):
        name = labelName(label)
        ref = TDF_Label()
        if self.shapes.IsReference_s(label) and self.shapes.GetReferredShape_s(label, ref):
            # instance: its own location, the product's geometry
            loc = loc.Multiplied(self.shapes.GetLocation_s(label))
            shape = self.shapes.GetShape_s(label)
            col = self.color(label, shape) or inherited
            self.walk(ref, loc, path + [name], col)
            return
        shape = self.shapes.GetShape_s(label)
        col = self.color(label, shape) or inherited
        if self.shapes.IsAssembly_s(label):
            kids = TDF_LabelSequence()
            self.shapes.GetComponents_s(label, kids, False)
            for i in range(1, kids.Length() + 1):
                self.walk(kids.Value(i), loc, path + [name] if not path or path[-1] != name else path, col)
            return
        self.addShape(shape.Moved(loc), path + [name], col)

    def addShape(self, shape, path, col):
        BRepMesh_IncrementalMesh(shape, self.deflection, False, 0.5, True)
        v0, t0, nFaces = self.nVerts, self.nTris, 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face(exp.Current())
            exp.Next()
            floc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, floc)
            if tri is None:
                continue
            trsf = floc.Transformation()
            n = tri.NbNodes()
            pts = np.empty((n, 3), np.float64)
            for k in range(1, n + 1):
                p = tri.Node(k).Transformed(trsf)
                pts[k - 1] = (p.X(), p.Y(), p.Z())
            idx = np.empty((tri.NbTriangles(), 3), np.int64)
            for k in range(1, tri.NbTriangles() + 1):
                idx[k - 1] = tri.Triangle(k).Get()
            idx -= 1
            if face.Orientation() == TopAbs_REVERSED:
                idx = idx[:, ::-1]
            # flat-ish normals per vertex from the triangles (good enough for shading and slope)
            fn = np.cross(pts[idx[:, 1]] - pts[idx[:, 0]], pts[idx[:, 2]] - pts[idx[:, 0]])
            vn = np.zeros_like(pts)
            for c in range(3):
                np.add.at(vn, idx[:, c], fn)
            vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
            self.verts.append(pts.astype(np.float32))
            self.norms.append(vn.astype(np.float32))
            self.tris.append((idx + self.nVerts).astype(np.uint32))
            self.nVerts += n
            self.nTris += len(idx)
            nFaces += 1
        if self.nVerts == v0:
            return
        vs = np.concatenate(self.verts[len(self.verts) - nFaces:])
        self.parts.append({'name': path[-1], 'path': path, 'color': col,
                           'vertexStart': v0, 'vertexCount': self.nVerts - v0,
                           'triangleStart': t0, 'triangleCount': self.nTris - t0,
                           'bboxMin': [round(float(x), 4) for x in vs.min(0)],
                           'bboxMax': [round(float(x), 4) for x in vs.max(0)]})

    def save(self, outDir):
        os.makedirs(outDir, exist_ok=True)
        np.savez_compressed(os.path.join(outDir, 'mesh.npz'), vertices=np.concatenate(self.verts),
                            normals=np.concatenate(self.norms), triangles=np.concatenate(self.tris))
        with open(os.path.join(outDir, 'parts.json'), 'w', encoding='utf-8') as f:
            json.dump(self.parts, f, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board', help='boards/<board> folder')
    ap.add_argument('--deflection', type=float, default=0.02, help='linear tessellation tolerance, mm')
    args = ap.parse_args()
    steps = glob.glob(os.path.join(args.board, 'export', 'ExportSTEP', '*.step'))
    if not steps:
        sys.exit(f'no STEP in {args.board}/export/ExportSTEP')
    t = time.time()
    m = StepMesher(steps[0], args.deflection)
    print(f'read {os.path.basename(steps[0])} in {time.time() - t:.1f} s')
    m.run()
    m.save(os.path.join(args.board, 'model'))
    print(f'{len(m.parts)} parts, {m.nVerts} vertices, {m.nTris} triangles, {time.time() - t:.1f} s')


if __name__ == '__main__':
    main()
