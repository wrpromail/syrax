import axios from "axios";

async function sendInferenceRequest(modelName, modelVersion, input) {
    const response = await axios.post(
      `http://localhost:8000/v2/models/${modelName}/versions/${modelVersion}/infer`,
      {
        id: "your_request_id",
        inputs: [
          {
            name: "input_name",
            datatype: "FP32",
            shape: [1, 3, 224, 224],
            data: input,
          },
        ],
        outputs: [
          {
            name: "output_name",
            parameters: {
              classification: 1,
            },
          },
        ],
      }
    );
  
    return response.data;
  }