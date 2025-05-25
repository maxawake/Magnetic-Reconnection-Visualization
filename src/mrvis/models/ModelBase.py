from __future__ import division
from paraview.util.vtkAlgorithm import *
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkStructuredGrid, vtkImageData
from vtkmodules.numpy_interface import dataset_adapter as dsa


class mrvisModelBase(VTKPythonAlgorithmBase):
    def __init__(self):
        self._dimensions = self._default_dimensions
        self._extent = self._default_extent
        self._output_field = self._enable_field_output
        num_outputs = 2 if self._enable_field_output else 1
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=num_outputs)

    def FillOutputPortInformation(self, port, info):
        if port == 1:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkImageData")
        elif port == 0:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkImageData")
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        executive = self.GetExecutive()

        dim_spatial = len(self._dimensions)
        if self._time_dependent:
            dim_spatial -= 1
        dim_spatial = min(3, dim_spatial)

        extent = [0] * 6
        for i in range(dim_spatial):
            extent[2 * i + 1] = self._dimensions[i] - 1
        image_origin = self._extent[: 2 * dim_spatial : 2]
        if len(image_origin) < 3:
            image_origin += [0.0] * (len(image_origin) - 3)
        image_spacing = [
            (self._extent[2 * i + 1] - self._extent[2 * i]) / (self._dimensions[i] - 1) for i in range(dim_spatial)
        ]
        if len(image_spacing) < 3:
            image_spacing += [0.0] * (len(image_spacing) - 3)

        out_info = outInfo.GetInformationObject(0)

        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        if self._time_dependent:
            time_steps = list(np.linspace(self._extent[-2], self._extent[-1], self._dimensions[-1]))

            out_info.Set(executive.TIME_RANGE(), [time_steps[0], time_steps[-1]], 2)
            out_info.Set(executive.TIME_STEPS(), time_steps, len(time_steps))

        out_info.Set(executive.WHOLE_EXTENT(), extent, 6)
        out_info.Set(vtkDataObject.ORIGIN(), image_origin, 3)
        out_info.Set(vtkDataObject.SPACING(), image_spacing, 3)

        if self._enable_field_output:
            out_info_1 = outInfo.GetInformationObject(1)
            out_info_1.Remove(executive.TIME_STEPS())
            out_info_1.Remove(executive.TIME_RANGE())

            if self._output_field:
                out_info_1.Set(executive.WHOLE_EXTENT(), extent, 6)
                out_info_1.Set(vtkDataObject.ORIGIN(), image_origin, 3)
                out_info_1.Set(vtkDataObject.SPACING(), image_spacing, 3)
            else:
                out_info_1.Set(executive.WHOLE_EXTENT(), [0, -1, 0, -1, 0, -1], 6)

        return 1

    def SetDimensions(self, *args):
        raise NotImplementedError()

    def SetExtent(self, *args):
        raise NotImplementedError()

    def Sample(self, *args):
        raise NotImplementedError()

    def RequestData(self, request, inInfo, outInfo):
        executive = self.GetExecutive()
        out_info = outInfo.GetInformationObject(0)
        image_output = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 0))

        time = 0.0
        if out_info.Has(executive.UPDATE_TIME_STEP()):
            time = out_info.Get(executive.UPDATE_TIME_STEP())
        time_index = 0
        if self._time_dependent:
            time_spacing = (self._extent[-1] - self._extent[-2]) / (self._dimensions[-1] - 1)
            time_origin = self._extent[-2]
            time_index = int(np.round((time - time_origin) / time_spacing))
            time_index = np.clip(time_index, 0, self._dimensions[-1] - 1)

        dim_spatial = len(self._dimensions)
        if self._time_dependent:
            dim_spatial -= 1
        dim_spatial = min(3, dim_spatial)

        coords = list(
            np.linspace(self._extent[2 * i], self._extent[2 * i + 1], self._dimensions[i])
            for i in range(len(self._dimensions))
        )

        if not self._output_field:
            for i in range(dim_spatial, len(self._dimensions)):
                coords[i] = np.array([self._extent[2 * i]])
            if self._time_dependent:
                coords[-1] = np.array([time])

        grid = np.meshgrid(*coords, indexing="ij")

        samples = self.Sample(*grid)
        if not isinstance(samples, dict):
            samples = {"u": samples}
        for n in samples:
            if type(samples[n]) is list or type(samples[n]) is tuple:
                samples[n] = np.stack(samples[n], axis=-1)
            if len(samples[n].shape) == len(self._dimensions):
                samples[n] = samples[n].reshape(tuple(self._dimensions) + (1,))

        if self._enable_field_output and self._output_field:
            field_output = dsa.WrapDataObject(vtkImageData.GetData(outInfo, 1))

            dimensions = np.array(self._dimensions)
            origin = np.array([a[0] for a in coords])
            spacing = np.array([abs(a[1] - a[0]) for a in coords])

            field_output.SetDimensions(*dimensions[:3])
            field_output.SetOrigin(*origin[:3])
            field_output.SetSpacing(*spacing[:3])

            field_output.FieldData.append(dimensions.reshape((1, dimensions.shape[0])), "Dimensions")
            field_output.FieldData.append(origin.reshape((1, origin.shape[0])), "Origin")
            field_output.FieldData.append(spacing.reshape((1, spacing.shape[0])), "Spacing")
            for n in ("Dimensions", "Origin", "Spacing"):
                image_output.VTKObject.GetFieldData().AddArray(field_output.VTKObject.GetFieldData().GetArray(n))
            for n, s in samples.items():
                field_output.FieldData.append(s.reshape((-1, s.shape[-1]), order="F"), n)
                image_output.VTKObject.GetFieldData().AddArray(field_output.VTKObject.GetFieldData().GetArray(n))

        # image data output
        image_dimensions = self._dimensions[:dim_spatial]
        if len(image_dimensions) < 3:
            image_dimensions += [1] * (3 - len(image_dimensions))
        image_origin = self._extent[: 2 * dim_spatial : 2]
        if len(image_origin) < 3:
            image_origin += [0.0] * (3 - len(image_origin))
        image_spacing = [coords[i][1] - coords[i][0] for i in range(dim_spatial)]
        if len(image_spacing) < 3:
            image_spacing += [0.0] * (3 - len(image_spacing))

        image_output.SetDimensions(image_dimensions)
        image_output.SetOrigin(image_origin)
        image_output.SetSpacing(image_spacing)

        time_slice = (slice(None),) * dim_spatial

        if self._time_dependent:
            time_slice += (0,) * (len(self._dimensions) - 1 - dim_spatial)
            if not self._output_field:
                time_slice += (0,)
            else:
                time_slice += (time_index,)
            time_slice += (slice(dim_spatial),)
        else:
            time_slice += (0,) * (len(self._dimensions) - dim_spatial)
            time_slice += (slice(None),)

        for n, s in samples.items():
            s_image = np.copy(s[time_slice])
            s_image = s_image.reshape((-1, s_image.shape[-1]), order="F")
            image_output.PointData.append(s_image, n)

            if s_image.shape[-1] == 2:
                u_vec = np.c_[s_image, np.zeros(s_image.shape[0])]
                image_output.PointData.append(u_vec.reshape((-1, u_vec.shape[-1]), order="F"), n + "_vec")

        if self._time_dependent:
            out_info.Set(vtkDataObject.DATA_TIME_STEP(), time)
            out_info.Set(vtkDataObject.ORIGIN(), image_origin, 3)
            out_info.Set(vtkDataObject.SPACING(), image_spacing, 3)

        extent = [0] * 6
        for i in range(dim_spatial):
            extent[2 * i + 1] = self._dimensions[i] - 1
        out_info.Set(executive.WHOLE_EXTENT(), extent, 6)

        return 1


def mrvis_model(dimensions, extent, time_dependent=False, ndfield=False, add_to_menu=True):
    def decorator(original_class):
        original_class._default_dimensions = dimensions
        original_class._default_extent = extent
        original_class._time_dependent = time_dependent
        original_class._enable_field_output = ndfield

        if len(extent) != 2 * len(dimensions):
            raise ValueError()

        # dynamically compile functions with len(dimensions) and len(extent) arguments
        d = {}
        arg_dimensions = ", ".join(list("a_{}".format(i) for i in range(len(dimensions))))
        arg_extent = ", ".join(list("a_{}".format(i) for i in range(len(extent))))
        exec("def SetDimensions(self, {0}): self._dimensions = [{0}]; self.Modified()".format(arg_dimensions), d)
        exec("def SetExtent(self, {0}): self._extent = [{0}]; self.Modified()".format(arg_extent), d)

        original_class.SetDimensions = smproperty.intvector(label="Dimensions", default_values=dimensions)(
            d["SetDimensions"]
        )
        original_class.SetExtent = smproperty.doublevector(label="Extent", default_values=extent)(d["SetExtent"])

        def SetOutputField(self, b):
            self._output_field = b
            self.Modified()

        xml = '<OutputPort name="Image" index="0" id="port0"/>\n'

        if original_class._enable_field_output:
            original_class.SetOutputField = smproperty.intvector(label="Output Field", default_values=True)(
                smdomain_boolean()(SetOutputField)
            )
            xml += '<OutputPort name="Field" index="1" id="port1"/>\n'

        original_class = smproperty.xml(
            xml
            + """
            <DoubleVectorProperty information_only="1"
                                  name="TimeRange">
              <TimeRangeInformationHelper />
            </DoubleVectorProperty>"""
        )(original_class)

        # if add_to_menu:
        #     original_class = smhint_menu("mrvis :: model")(original_class)

        return original_class

    return decorator
