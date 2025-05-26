import paraview.detail.pythonalgorithm as pvdetail
from paraview.util.vtkAlgorithm import smproperty


def smproperty_inputarray(
    label,
    idx=0,
    input_domain_name="input_array",
    input_name="Input",
    none_string=None,
    attribute_type="Scalars",
    **kwargs,
):
    def generate(func, attrs):
        xml = """<StringVectorProperty
                        name="SelectInputScalars{idx}"
                        label="{label}"
                        command="{command}"
                        default_values="{idx}"
                        number_of_elements="5"
                        element_types="0 0 0 0 2"
                        animateable="0">
                        <ArrayListDomain
                          name="array_list"
                          attribute_type="{attribute_type}"
                          input_domain_name="{input_domain_name}"
                          {none_property}>
                          <RequiredProperties>
                            <Property
                              name="{input_name}"
                              function="Input" />
                          </RequiredProperties>
                        </ArrayListDomain>
                      </StringVectorProperty>
            """.format(**attrs)
        smproperty._append_xml(func, xml)

    attrs = {
        "idx": idx,
        "label": label,
        "input_domain_name": input_domain_name,
        "input_name": input_name,
        "none_property": 'none_string="{}"'.format(none_string) if none_string else "",
        "attribute_type": attribute_type,
    }
    attrs.update(kwargs)
    return pvdetail._create_decorator(
        attrs, update_func=smproperty._update_property_defaults, generate_xml_func=generate
    )
