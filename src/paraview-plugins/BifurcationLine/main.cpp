#include <vtkNew.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>
#include "MyAlgorithm.h"

int main() {
  vtkNew<vtkXMLPolyDataReader> reader;
  vtkNew<MyAlgorithm> filter;
  vtkNew<vtkXMLPolyDataWriter> writer;

  reader->SetFileName("input.vtp");
  filter->SetInputConnection(reader->GetOutputPort());
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName("output.vtp");

  writer->Update();
  return 0;
}
