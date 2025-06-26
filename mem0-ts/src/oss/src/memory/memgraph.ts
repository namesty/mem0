const BM25Vectorizer = require('wink-nlp/utilities/bm25-vectorizer');
const model = require('wink-eng-lite-web-model');
const nlp = require('wink-nlp' )(model);

interface Config {
  graph_store: {
    config: {
      url: string;
      username: string;
      password: string;
    };
    llm?: {
      provider: string;
    };
    custom_prompt?: string;
  };
  embedder: {
    provider: string;
    config: {
      embedding_dims: number;
      [key: string]: any;
    };
  };
  llm: {
    provider: string;
    config: any;
  };
}

interface Filters {
  user_id: string;
  [key: string]: any;
}

interface EntityTypeMap {
  [key: string]: string;
}

interface Entity {
  source: string;
  relationship: string;
  destination: string;
}

interface SearchResult {
  source: string;
  source_id?: number;
  relationship: string;
  relation_id?: number;
  destination: string;
  destination_id?: number;
  target?: string;
  similarity?: number;
}

interface DeleteResult {
  source: string;
  target: string;
  relationship: string;
}

interface AddResult {
  deleted_entities: any[];
  added_entities: any[];
}

interface ToolCall {
  name: string;
  arguments: any;
}

interface LLMResponse {
  tool_calls: ToolCall[];
}

export const EXTRACT_ENTITIES_TOOL = {
  name: "extract_entities",
  description: "Extract entities and their types from the given text",
  parameters: {
    type: "object",
    properties: {
      entities: {
        type: "array",
        description: "List of entities found in the text",
        items: {
          type: "object",
          properties: {
            entity: {
              type: "string",
              description: "The entity name"
            },
            entity_type: {
              type: "string",
              description: "The type of the entity (e.g., person, organization, location, etc.)"
            }
          },
          required: ["entity", "entity_type"]
        }
      }
    },
    required: ["entities"]
  }
};

// Structured tool format (for OpenAI structured output / Azure OpenAI structured)
export const EXTRACT_ENTITIES_STRUCT_TOOL = {
  type: "function",
  function: {
    name: "extract_entities",
    description: "Extract entities and their types from the given text",
    parameters: {
      type: "object",
      properties: {
        entities: {
          type: "array",
          description: "List of entities found in the text",
          items: {
            type: "object",
            properties: {
              entity: {
                type: "string",
                description: "The entity name"
              },
              entity_type: {
                type: "string",
                description: "The type of the entity (e.g., person, organization, location, etc.)"
              }
            },
            required: ["entity", "entity_type"]
          }
        }
      },
      required: ["entities"]
    }
  }
};

export const RELATIONS_TOOL = {
  name: "extract_relationships",
  description: "Extract relationships between entities in the text",
  parameters: {
    type: "object",
    properties: {
      entities: {
        type: "array",
        description: "List of relationships between entities",
        items: {
          type: "object",
          properties: {
            source: {
              type: "string",
              description: "The source entity in the relationship"
            },
            relationship: {
              type: "string",
              description: "The type of relationship between source and destination"
            },
            destination: {
              type: "string",
              description: "The destination entity in the relationship"
            }
          },
          required: ["source", "relationship", "destination"]
        }
      }
    },
    required: ["entities"]
  }
};

export const RELATIONS_STRUCT_TOOL = {
  type: "function",
  function: {
    name: "extract_relationships",
    description: "Extract relationships between entities in the text",
    parameters: {
      type: "object",
      properties: {
        entities: {
          type: "array",
          description: "List of relationships between entities",
          items: {
            type: "object",
            properties: {
              source: {
                type: "string",
                description: "The source entity in the relationship"
              },
              relationship: {
                type: "string",
                description: "The type of relationship between source and destination"
              },
              destination: {
                type: "string",
                description: "The destination entity in the relationship"
              }
            },
            required: ["source", "relationship", "destination"]
          }
        }
      },
      required: ["entities"]
    }
  }
};

export const DELETE_MEMORY_TOOL_GRAPH = {
  name: "delete_graph_memory",
  description: "Delete outdated or incorrect relationships from the graph",
  parameters: {
    type: "object",
    properties: {
      source: {
        type: "string",
        description: "The source entity of the relationship to delete"
      },
      relationship: {
        type: "string",
        description: "The type of relationship to delete"
      },
      destination: {
        type: "string",
        description: "The destination entity of the relationship to delete"
      }
    },
    required: ["source", "relationship", "destination"]
  }
};

export const DELETE_MEMORY_STRUCT_TOOL_GRAPH = {
  type: "function",
  function: {
    name: "delete_graph_memory",
    description: "Delete outdated or incorrect relationships from the graph",
    parameters: {
      type: "object",
      properties: {
        source: {
          type: "string",
          description: "The source entity of the relationship to delete"
        },
        relationship: {
          type: "string",
          description: "The type of relationship to delete"
        },
        destination: {
          type: "string",
          description: "The destination entity of the relationship to delete"
        }
      },
      required: ["source", "relationship", "destination"]
    }
  }
};

// Utility functions (these would be imported from utils modules)
const EXTRACT_RELATIONS_PROMPT = `
  You are a smart assistant who understands relationships between entities.
  Extract relationships from the given text.
  USER_ID: {USER_ID}
  CUSTOM_PROMPT: {CUSTOM_PROMPT}
`;

function formatEntities(searchOutput: SearchResult[]): string {
  // Implementation to format entities as string
  return searchOutput.map(item => 
    `${item.source} -[${item.relationship}]-> ${item.destination}`
  ).join('\n');
}

function getDeleteMessages(searchOutputString: string, data: string, userId: string): [string, string] {
  // Implementation to generate system and user prompts for deletion
  const systemPrompt = `You are an assistant that identifies outdated relationships to delete.`;
  const userPrompt = `Current relationships:\n${searchOutputString}\n\nNew data: ${data}\n\nUser ID: ${userId}`;
  return [systemPrompt, userPrompt];
}

// Mock implementations for external dependencies
class Memgraph {
  constructor(url: string, username: string, password: string) {
    // Initialize connection
  }

  async query(cypher: string, params: any = {}): Promise<any[]> {
    // Execute Cypher query
    return [];
  }
}

class EmbedderFactory {
  static create(provider: string, config: any, options: any): any {
    return {
      embed: async (text: string): Promise<number[]> => {
        // Return embedding vector
        return [];
      }
    };
  }
}

class LlmFactory {
  static create(provider: string, config: any): any {
    return {
      generateResponse: async (params: { messages: any[], tools: any[] }): Promise<LLMResponse> => {
        // Generate LLM response
        return { tool_calls: [] };
      }
    };
  }
}

class MemoryGraph {
  private config: Config;
  private graph: Memgraph;
  private embeddingModel: any;
  private llmProvider: string;
  private llm: any;
  private userId: string | null = null;
  private threshold: number = 0.7;

  constructor(config: Config) {
    this.config = config;
    this.graph = new Memgraph(
      this.config.graph_store.config.url,
      this.config.graph_store.config.username,
      this.config.graph_store.config.password
    );

    this.embeddingModel = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config,
      { enableEmbeddings: true }
    );

    this.llmProvider = "openai_structured";
    if (this.config.llm.provider) {
      this.llmProvider = this.config.llm.provider;
    }
    if (this.config.graph_store.llm) {
      this.llmProvider = this.config.graph_store.llm.provider;
    }

    this.llm = LlmFactory.create(this.llmProvider, this.config.llm.config);

    // Setup Memgraph indices
    this.setupMemgraph();
  }

  private async setupMemgraph(): Promise<void> {
    const embeddingDims = this.config.embedder.config.embedding_dims;
    
    // Create vector index
    const createVectorIndexQuery = `CREATE VECTOR INDEX memzero ON :Entity(embedding) WITH CONFIG {'dimension': ${embeddingDims}, 'capacity': 1000, 'metric': 'cos'};`;
    await this.graph.query(createVectorIndexQuery);
    
    // Create label property index
    const createLabelPropIndexQuery = `CREATE INDEX ON :Entity(user_id);`;
    await this.graph.query(createLabelPropIndexQuery);
    
    // Create label index
    const createLabelIndexQuery = `CREATE INDEX ON :Entity;`;
    await this.graph.query(createLabelIndexQuery);
  }

  async add(data: string, filters: Filters): Promise<AddResult> {
    const entityTypeMap = await this.retrieveNodesFromData(data, filters);
    const toBeAdded = await this.establishNodesRelationsFromData(data, filters, entityTypeMap);
    const searchOutput = await this.searchGraphDb(
      Object.keys(entityTypeMap),
      filters,
    );
    const toBeDeleted = await this.getDeleteEntitiesFromSearchOutput(
      searchOutput,
      data,
      filters
    );

    const deletedEntities = await this.deleteEntities(toBeDeleted, filters.user_id);
    const addedEntities = await this.addEntities(
      toBeAdded,
      filters.user_id,
      entityTypeMap
    );

    return { deleted_entities: deletedEntities, added_entities: addedEntities };
  }

  async search(query: string, filters: Filters, limit: number = 100): Promise<SearchResult[]> {
    const entityTypeMap = await this.retrieveNodesFromData(query, filters);
    const searchOutput = await this.searchGraphDb(
      Object.keys(entityTypeMap),
      filters
    );

    if (!searchOutput.length) {
      return [];
    }

    // Prepare documents for BM25 ranking
    const documents = searchOutput.map(item => 
      `${item.source} ${item.relationship} ${item.destination}`
    );

    const bm25 = BM25Vectorizer() as any;
    documents.forEach(doc => {
      bm25.learn(nlp.readDoc(doc).tokens().out());
    });
    bm25.consolidate();
    
    const queryTokens = nlp.readDoc(query).tokens().out();
    const scores = documents.map((_, index) => ({
      index,
      score: bm25.scoreOf(queryTokens, index)
    }));
    
    const topResults = scores
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(result => searchOutput[result.index]);

    const searchResults: SearchResult[] = topResults.map(item => ({
      source: item.source,
      relationship: item.relationship,
      destination: item.destination
    }));

    console.log(`Returned ${searchResults.length} search results`);
    return searchResults;
  }

  async deleteAll(filters: Filters): Promise<void> {
    const cypher = `
      MATCH (n {user_id: $user_id})
      DETACH DELETE n
    `;
    const params = { user_id: filters.user_id };
    await this.graph.query(cypher, params);
  }

  async getAll(filters: Filters, limit: number = 100): Promise<SearchResult[]> {
    const query = `
      MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id})
      RETURN n.name AS source, type(r) AS relationship, m.name AS target
      LIMIT $limit
    `;
    const results = await this.graph.query(query, {
      user_id: filters.user_id,
      limit: limit
    });

    const finalResults: SearchResult[] = results.map(result => ({
      source: result.source,
      relationship: result.relationship,
      target: result.target,
      destination: result.target
    }));

    console.log(`Retrieved ${finalResults.length} relationships`);
    return finalResults;
  }

  private async retrieveNodesFromData(data: string, filters: Filters): Promise<EntityTypeMap> {
    const tools = this.llmProvider in ["azure_openai_structured", "openai_structured"]
      ? [EXTRACT_ENTITIES_STRUCT_TOOL]
      : [EXTRACT_ENTITIES_TOOL];

    const searchResults = await this.llm.generateResponse({
      messages: [
        {
          role: "system",
          content: `You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use ${filters.user_id} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.`
        },
        {
          role: "user",
          content: data
        }
      ],
      tools: tools
    });

    const entityTypeMap: EntityTypeMap = {};

    try {
      for (const toolCall of searchResults.tool_calls) {
        if (toolCall.name !== "extract_entities") continue;
        
        for (const item of toolCall.arguments.entities) {
          entityTypeMap[item.entity] = item.entity_type;
        }
      }
    } catch (error) {
      console.error(`Error in search tool: ${error}, llm_provider=${this.llmProvider}`);
    }

    // Normalize entity names and types
    const normalizedMap: EntityTypeMap = {};
    for (const [key, value] of Object.entries(entityTypeMap)) {
      const normalizedKey = key.toLowerCase().replace(/ /g, "_");
      const normalizedValue = value.toLowerCase().replace(/ /g, "_");
      normalizedMap[normalizedKey] = normalizedValue;
    }

    console.debug(`Entity type map: ${JSON.stringify(normalizedMap)}`);
    return normalizedMap;
  }

  private async establishNodesRelationsFromData(
    data: string,
    filters: Filters,
    entityTypeMap: EntityTypeMap
  ): Promise<Entity[]> {
    let messages: any[];

    if (this.config.graph_store.custom_prompt) {
      messages = [
        {
          role: "system",
          content: EXTRACT_RELATIONS_PROMPT
            .replace("USER_ID", filters.user_id)
            .replace("CUSTOM_PROMPT", `4. ${this.config.graph_store.custom_prompt}`)
        },
        {
          role: "user",
          content: data
        }
      ];
    } else {
      messages = [
        {
          role: "system",
          content: EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters.user_id)
        },
        {
          role: "user",
          content: `List of entities: ${Object.keys(entityTypeMap)}. \n\nText: ${data}`
        }
      ];
    }

    const tools = this.llmProvider in ["azure_openai_structured", "openai_structured"]
      ? [RELATIONS_STRUCT_TOOL]
      : [RELATIONS_TOOL];

    const extractedEntities = await this.llm.generateResponse({
      messages: messages,
      tools: tools
    });

    let entities: Entity[] = [];
    if (extractedEntities.tool_calls.length > 0) {
      entities = extractedEntities.tool_calls[0].arguments.entities;
    }

    entities = this.removeSpacesFromEntities(entities);
    console.debug(`Extracted entities: ${JSON.stringify(entities)}`);
    return entities;
  }

  private async searchGraphDb(
    nodeList: string[],
    filters: Filters,
    limit: number = 100
  ): Promise<SearchResult[]> {
    const resultRelations: SearchResult[] = [];

    for (const node of nodeList) {
      const nEmbedding = await this.embeddingModel.embed(node);

      const cypherQuery = `
        MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity)
        WHERE n.embedding IS NOT NULL
        WITH collect(n) AS nodes1, collect(m) AS nodes2, r
        CALL node_similarity.cosine_pairwise("embedding", nodes1, nodes2)
        YIELD node1, node2, similarity
        WITH node1, node2, similarity, r
        WHERE similarity >= $threshold
        RETURN node1.user_id AS source, id(node1) AS source_id, type(r) AS relationship, 
               id(r) AS relation_id, node2.user_id AS destination, id(node2) AS destination_id, similarity
        UNION
        MATCH (n:Entity {user_id: $user_id})<-[r]-(m:Entity)
        WHERE n.embedding IS NOT NULL
        WITH collect(n) AS nodes1, collect(m) AS nodes2, r
        CALL node_similarity.cosine_pairwise("embedding", nodes1, nodes2)
        YIELD node1, node2, similarity
        WITH node1, node2, similarity, r
        WHERE similarity >= $threshold
        RETURN node2.name AS source, id(node2) AS source_id, type(r) AS relationship, 
               id(r) AS relation_id, node1.name AS destination, id(node1) AS destination_id, similarity
        ORDER BY similarity DESC
        LIMIT $limit;
      `;

      const params = {
        n_embedding: nEmbedding,
        threshold: this.threshold,
        user_id: filters.user_id,
        limit: limit
      };

      const ans = await this.graph.query(cypherQuery, params);
      resultRelations.push(...ans);
    }

    return resultRelations;
  }

  private async getDeleteEntitiesFromSearchOutput(
    searchOutput: SearchResult[],
    data: string,
    filters: Filters
  ): Promise<Entity[]> {
    const searchOutputString = formatEntities(searchOutput);
    const [systemPrompt, userPrompt] = getDeleteMessages(
      searchOutputString,
      data,
      filters.user_id
    );

    const tools = this.llmProvider in ["azure_openai_structured", "openai_structured"]
      ? [DELETE_MEMORY_STRUCT_TOOL_GRAPH]
      : [DELETE_MEMORY_TOOL_GRAPH];

    const memoryUpdates = await this.llm.generateResponse({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      tools: tools
    });

    const toBeDeleted: Entity[] = [];
    for (const item of memoryUpdates.tool_calls) {
      if (item.name === "delete_graph_memory") {
        toBeDeleted.push(item.arguments);
      }
    }

    const normalizedDeleted = this.removeSpacesFromEntities(toBeDeleted);
    console.debug(`Deleted relationships: ${JSON.stringify(normalizedDeleted)}`);
    return normalizedDeleted;
  }

  private async deleteEntities(toBeDeleted: Entity[], userId: string): Promise<DeleteResult[]> {
    const results: DeleteResult[] = [];

    for (const item of toBeDeleted) {
      const { source, destination, relationship } = item;

      const cypher = `
        MATCH (n:Entity {name: $source_name, user_id: $user_id})
        -[r:${relationship}]->
        (m {name: $dest_name, user_id: $user_id})
        DELETE r
        RETURN 
          n.name AS source,
          m.name AS target,
          type(r) AS relationship
      `;

      const params = {
        source_name: source,
        dest_name: destination,
        user_id: userId
      };

      const result = await this.graph.query(cypher, params);
      results.push(...result);
    }

    return results;
  }

  private async addEntities(
    toBeAdded: Entity[],
    userId: string,
    entityTypeMap: EntityTypeMap
  ): Promise<any[]> {
    const results: any[] = [];

    for (const item of toBeAdded) {
      const { source, destination, relationship } = item;
      
      // Get entity types
      const sourceType = entityTypeMap[source] || "unknown";
      const destinationType = entityTypeMap[destination] || "unknown";

      // Generate embeddings
      const sourceEmbedding = await this.embeddingModel.embed(source);
      const destEmbedding = await this.embeddingModel.embed(destination);

      // Search for existing nodes
      const sourceNodeSearchResult = await this.searchSourceNode(
        sourceEmbedding,
        userId,
        0.9
      );
      const destinationNodeSearchResult = await this.searchDestinationNode(
        destEmbedding,
        userId,
        0.9
      );

      let cypher: string;
      let params: any;

      if (!destinationNodeSearchResult.length && sourceNodeSearchResult.length) {
        cypher = `
          MATCH (source:Entity)
          WHERE id(source) = $source_id
          MERGE (destination:${destinationType}:Entity {name: $destination_name, user_id: $user_id})
          ON CREATE SET
            destination.created = timestamp(),
            destination.embedding = $destination_embedding,
            destination:Entity
          MERGE (source)-[r:${relationship}]->(destination)
          ON CREATE SET 
            r.created = timestamp()
          RETURN source.name AS source, type(r) AS relationship, destination.name AS target
        `;

        params = {
          source_id: sourceNodeSearchResult[0]["id(source_candidate)"],
          destination_name: destination,
          destination_embedding: destEmbedding,
          user_id: userId
        };
      } else if (destinationNodeSearchResult.length && !sourceNodeSearchResult.length) {
        cypher = `
          MATCH (destination:Entity)
          WHERE id(destination) = $destination_id
          MERGE (source:${sourceType}:Entity {name: $source_name, user_id: $user_id})
          ON CREATE SET
            source.created = timestamp(),
            source.embedding = $source_embedding,
            source:Entity
          MERGE (source)-[r:${relationship}]->(destination)
          ON CREATE SET 
            r.created = timestamp()
          RETURN source.name AS source, type(r) AS relationship, destination.name AS target
        `;

        params = {
          destination_id: destinationNodeSearchResult[0]["id(destination_candidate)"],
          source_name: source,
          source_embedding: sourceEmbedding,
          user_id: userId
        };
      } else if (sourceNodeSearchResult.length && destinationNodeSearchResult.length) {
        cypher = `
          MATCH (source:Entity)
          WHERE id(source) = $source_id
          MATCH (destination:Entity)
          WHERE id(destination) = $destination_id
          MERGE (source)-[r:${relationship}]->(destination)
          ON CREATE SET 
            r.created_at = timestamp(),
            r.updated_at = timestamp()
          RETURN source.name AS source, type(r) AS relationship, destination.name AS target
        `;

        params = {
          source_id: sourceNodeSearchResult[0]["id(source_candidate)"],
          destination_id: destinationNodeSearchResult[0]["id(destination_candidate)"],
          user_id: userId
        };
      } else {
        cypher = `
          MERGE (n:${sourceType}:Entity {name: $source_name, user_id: $user_id})
          ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding, n:Entity
          ON MATCH SET n.embedding = $source_embedding
          MERGE (m:${destinationType}:Entity {name: $dest_name, user_id: $user_id})
          ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding, m:Entity
          ON MATCH SET m.embedding = $dest_embedding
          MERGE (n)-[rel:${relationship}]->(m)
          ON CREATE SET rel.created = timestamp()
          RETURN n.name AS source, type(rel) AS relationship, m.name AS target
        `;

        params = {
          source_name: source,
          dest_name: destination,
          source_embedding: sourceEmbedding,
          dest_embedding: destEmbedding,
          user_id: userId
        };
      }

      const result = await this.graph.query(cypher, params);
      results.push(...result);
    }

    return results;
  }

  private removeSpacesFromEntities(entityList: Entity[]): Entity[] {
    return entityList.map(item => ({
      source: item.source.toLowerCase().replace(/ /g, "_"),
      relationship: item.relationship.toLowerCase().replace(/ /g, "_"),
      destination: item.destination.toLowerCase().replace(/ /g, "_")
    }));
  }

  private async searchSourceNode(
    sourceEmbedding: number[],
    userId: string,
    threshold: number = 0.9
  ): Promise<any[]> {
    const cypher = `
      CALL vector_search.search("memzero", 1, $source_embedding) 
      YIELD distance, node, similarity
      WITH node AS source_candidate, similarity
      WHERE source_candidate.user_id = $user_id AND similarity >= $threshold
      RETURN id(source_candidate);
    `;

    const params = {
      source_embedding: sourceEmbedding,
      user_id: userId,
      threshold: threshold
    };

    return await this.graph.query(cypher, params);
  }

  private async searchDestinationNode(
    destinationEmbedding: number[],
    userId: string,
    threshold: number = 0.9
  ): Promise<any[]> {
    const cypher = `
      CALL vector_search.search("memzero", 1, $destination_embedding) 
      YIELD distance, node, similarity
      WITH node AS destination_candidate, similarity
      WHERE node.user_id = $user_id AND similarity >= $threshold
      RETURN id(destination_candidate);
    `;

    const params = {
      destination_embedding: destinationEmbedding,
      user_id: userId,
      threshold: threshold
    };

    return await this.graph.query(cypher, params);
  }
}