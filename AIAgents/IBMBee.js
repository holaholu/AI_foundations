import {BeeAgent} from "bee-agent-framework/agents/bee/agent";
import {OllamaChatLLM} from "bee-agent-framework/adapters/ollama/chat";
import {TokenMemory} from "bee-agent-framework/memory/tokenMemory";
import {DuckDuckGoSearchTool} from "bee-agent-framework/tools/search/duckDuckGoSearch";
import {OpenMeteoTool} from "bee-agent-framework/tools/weather/openMeteo";
//Make sure to install the bee-agent-framework package



const llm = new OllamaChatLLM();
const memory = new TokenMemory({llm});
const tools = [new DuckDuckGoSearchTool(), new OpenMeteoTool()];
const agent = new BeeAgent({llm, memory, tools});

agent.observe((emitter) => {
  emitter.on("update", async ({ data }) => {
    console.log(`Update:`, data);
  });
});  

//To add a custom Tool
import {tool} from "bee-agent-framework/tools/base";

const customTool = tool("CustomTool", {
    description: "This is a custom tool for a specific task",
    run: async (input) => {
        return `Handled task with input: ${input}`;
    }
})

agent.addTool(customTool);


const response = await agent.run({prompt:"What is the weather like in New York?"}).observe((emitter) => {
    emitter.on("update", async({data,update,meta})   => {
        console.log(`Agent (${update.key}):`, update.value);
    })
})  


console.log(`Agent Response:`, response.result.text)




