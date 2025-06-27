import neo4j, { Driver } from "neo4j-driver";
import { EmbedderFactory, LLMFactory } from "../utils/factory";
import { MemoryConfig } from "../types";
import { BM25 } from "../utils/bm25";
import { logger } from "../utils/logger";

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
  source_id?: string;
  relationship: string;
  relation_id?: string;
  destination: string;
  destination_id?: string;
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
              description: "The entity name",
            },
            entity_type: {
              type: "string",
              description:
                "The type of the entity (e.g., person, organization, location, etc.)",
            },
          },
          required: ["entity", "entity_type"],
        },
      },
    },
    required: ["entities"],
  },
};

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
                description: "The entity name",
              },
              entity_type: {
                type: "string",
                description:
                  "The type of the entity (e.g., person, organization, location, etc.)",
              },
            },
            required: ["entity", "entity_type"],
          },
        },
      },
      required: ["entities"],
    },
  },
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
              description: "The source entity in the relationship",
            },
            relationship: {
              type: "string",
              description:
                "The type of relationship between source and destination",
            },
            destination: {
              type: "string",
              description: "The destination entity in the relationship",
            },
          },
          required: ["source", "relationship", "destination"],
        },
      },
    },
    required: ["entities"],
  },
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
                description: "The source entity in the relationship",
              },
              relationship: {
                type: "string",
                description:
                  "The type of relationship between source and destination",
              },
              destination: {
                type: "string",
                description: "The destination entity in the relationship",
              },
            },
            required: ["source", "relationship", "destination"],
          },
        },
      },
      required: ["entities"],
    },
  },
};

export const DELETE_MEMORY_TOOL_GRAPH = {
  name: "delete_graph_memory",
  description: "Delete outdated or incorrect relationships from the graph",
  parameters: {
    type: "object",
    properties: {
      source: {
        type: "string",
        description: "The source entity of the relationship to delete",
      },
      relationship: {
        type: "string",
        description: "The type of relationship to delete",
      },
      destination: {
        type: "string",
        description: "The destination entity of the relationship to delete",
      },
    },
    required: ["source", "relationship", "destination"],
  },
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
          description: "The source entity of the relationship to delete",
        },
        relationship: {
          type: "string",
          description: "The type of relationship to delete",
        },
        destination: {
          type: "string",
          description: "The destination entity of the relationship to delete",
        },
      },
      required: ["source", "relationship", "destination"],
    },
  },
};

const EXTRACT_RELATIONS_PROMPT = `
  You are a smart assistant who understands relationships between entities.
  Extract relationships from the given text.
  USER_ID: {USER_ID}
  CUSTOM_PROMPT: {CUSTOM_PROMPT}
`;

function formatEntities(searchOutput: SearchResult[]): string {
  return searchOutput
    .map(
      (item) => `${item.source} -[${item.relationship}]-> ${item.destination}`,
    )
    .join("\n");
}

function getDeleteMessages(
  searchOutputString: string,
  data: string,
  userId: string,
): [string, string] {
  const systemPrompt = `You are an assistant that identifies outdated relationships to delete.`;
  const userPrompt = `Current relationships:\n${searchOutputString}\n\nNew data: ${data}\n\nUser ID: ${userId}`;
  return [systemPrompt, userPrompt];
}

export class MemoryGraph {
  private config: MemoryConfig;
  private graph: Driver;
  private embeddingModel: any;
  private llmProvider: string;
  private llm: any;
  private threshold: number = 0.7;

  constructor(config: MemoryConfig) {
    this.config = config;
    if (
      !config.graphStore?.config?.url ||
      !config.graphStore?.config?.username ||
      !config.graphStore?.config?.password
    ) {
      throw new Error("Neo4j configuration is incomplete");
    }

    this.graph = neo4j.driver(
      config.graphStore.config.url,
      neo4j.auth.basic(
        config.graphStore.config.username,
        config.graphStore.config.password,
      ),
    );

    this.embeddingModel = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config,
    );

    this.llmProvider = "openai_structured";
    if (this.config.llm.provider) {
      this.llmProvider = this.config.llm.provider;
    }
    if (this.config.graphStore?.llm && this.config.graphStore.llm.provider) {
      this.llmProvider = this.config.graphStore.llm.provider;
    }

    this.llm = LLMFactory.create(this.llmProvider, this.config.llm.config);

    this.setupMemgraph();
  }

  private async setupMemgraph(): Promise<void> {
    if (!this.config.embedder.config.dimension) {
      throw new Error("'dimension' required in 'embedder' config for memgraph");
    }

    const embeddingDims = this.config.embedder.config.dimension;
    const queries = [
      `CREATE VECTOR INDEX memzero ON :Entity(embedding) WITH CONFIG {'dimension': ${embeddingDims}, 'capacity': 1000, 'metric': 'cos'};`,
      `CREATE INDEX ON :Entity(user_id);`,
      `CREATE INDEX ON :Entity;`,
    ];

    const session = this.graph.session();
    try {
      for (const q of queries) {
        await session.run(q);
      }
    } finally {
      await session.close();
    }
  }

  async add(data: string, filters: Filters): Promise<AddResult> {
    const entityTypeMap = await this.retrieveNodesFromData(data, filters);
    const toBeAdded = await this.establishNodesRelationsFromData(
      data,
      filters,
      entityTypeMap,
    );
    const searchOutput = await this.searchGraphDb(
      Object.keys(entityTypeMap),
      filters,
    );
    const toBeDeleted = await this.getDeleteEntitiesFromSearchOutput(
      searchOutput,
      data,
      filters,
    );

    const deletedEntities = await this.deleteEntities(
      toBeDeleted,
      filters.user_id,
    );
    const addedEntities = await this.addEntities(
      toBeAdded,
      filters.user_id,
      entityTypeMap,
    );

    return { deleted_entities: deletedEntities, added_entities: addedEntities };
  }

  async search(query: string, filters: Filters, limit: number = 100): Promise<SearchResult[]> {
    const entityTypeMap = await this.retrieveNodesFromData(query, filters);
    const searchOutput = await this.searchGraphDb(
      Object.keys(entityTypeMap),
      filters,
    );
    
    if (!searchOutput.length) {
      return [];
    }
    
    const searchOutputsSequence = searchOutput.map((item) => [
      item.source,
      item.relationship,
      item.destination,
    ]);

    const bm25 = new BM25(searchOutputsSequence);
    const tokenizedQuery = query.split(" ");
    const rerankedResults = bm25.search(tokenizedQuery).slice(0, 5);

    const searchResults = rerankedResults.map((item) => ({
      source: item[0],
      relationship: item[1],
      destination: item[2],
    }));

    logger.info(`Returned ${searchResults.length} search results`);
    return searchResults;
  }

  async deleteAll(filters: Filters): Promise<void> {
    const session = this.graph.session();
    try {
      await session.run(
        `MATCH (n {user_id: $user_id}) DETACH DELETE n`,
        { user_id: filters.user_id },
      );
    } finally {
      await session.close();
    }
  }

  async getAll(filters: Filters, limit: number = 100): Promise<SearchResult[]> {
    const session = this.graph.session();
    try {
      const result = await session.run(
        `MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id}) RETURN n.name AS source, type(r) AS relationship, m.name AS target LIMIT toInteger($limit)`,
        { user_id: filters.user_id, limit: Math.floor(Number(limit)) },
      );

      const finalResults: SearchResult[] = result.records.map((record) => ({
        source: record.get("source"),
        relationship: record.get("relationship"),
        target: record.get("target"),
        destination: record.get("target"),
      }));

      return finalResults;
    } finally {
      await session.close();
    }
  }

  private async retrieveNodesFromData(data: string, filters: Filters): Promise<EntityTypeMap> {
    const tools =
      this.llmProvider === "azure_openai_structured" || this.llmProvider === "openai_structured"
        ? [EXTRACT_ENTITIES_STRUCT_TOOL]
        : [EXTRACT_ENTITIES_TOOL];

    const searchResults: LLMResponse = await this.llm.generateResponse({
      messages: [
        {
          role: "system",
          content: `You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use ${filters.user_id} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.`,
        },
        { role: "user", content: data },
      ],
      tools,
    });

    const entityTypeMap: EntityTypeMap = {};
    try {
      for (const call of searchResults.tool_calls) {
        if (call.name !== "extract_entities") continue;
        for (const item of call.arguments.entities) {
          entityTypeMap[item.entity] = item.entity_type;
        }
      }
    } catch (_) {}

    const normalizedMap: EntityTypeMap = {};
    for (const [k, v] of Object.entries(entityTypeMap)) {
      normalizedMap[k.toLowerCase().replace(/ /g, "_")] = v
        .toLowerCase()
        .replace(/ /g, "_");
    }
    return normalizedMap;
  }

  private async establishNodesRelationsFromData(
    data: string,
    filters: Filters,
    entityTypeMap: EntityTypeMap,
  ): Promise<Entity[]> {
    let messages: any[];
    if (this.config.graphStore?.customPrompt) {
      messages = [
        {
          role: "system",
          content: EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters.user_id).replace(
            "CUSTOM_PROMPT",
            `4. ${this.config.graphStore.customPrompt}`,
          ),
        },
        { role: "user", content: data },
      ];
    } else {
      messages = [
        {
          role: "system",
          content: EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters.user_id),
        },
        {
          role: "user",
          content: `List of entities: ${Object.keys(entityTypeMap)}. \n\nText: ${data}`,
        },
      ];
    }

    const tools =
      this.llmProvider === "azure_openai_structured" || this.llmProvider === "openai_structured"
        ? [RELATIONS_STRUCT_TOOL]
        : [RELATIONS_TOOL];

    const extractedEntities: LLMResponse = await this.llm.generateResponse({
      messages,
      tools,
    });

    let entities: Entity[] = [];
    if (extractedEntities.tool_calls.length) {
      entities = extractedEntities.tool_calls[0].arguments.entities;
    }

    entities = this.removeSpacesFromEntities(entities);
    return entities;
  }

  private async searchGraphDb(
    nodeList: string[],
    filters: Filters,
    limit: number = 100,
  ): Promise<SearchResult[]> {
    const resultRelations: SearchResult[] = [];
    const session = this.graph.session();
    try {
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
          RETURN node1.name AS source, id(node1) AS source_id, type(r) AS relationship, id(r) AS relation_id, node2.name AS destination, id(node2) AS destination_id, similarity
          UNION
          MATCH (n:Entity {user_id: $user_id})<-[r]-(m:Entity)
          WHERE n.embedding IS NOT NULL
          WITH collect(n) AS nodes1, collect(m) AS nodes2, r
          CALL node_similarity.cosine_pairwise("embedding", nodes1, nodes2)
          YIELD node1, node2, similarity
          WITH node1, node2, similarity, r
          WHERE similarity >= $threshold
          RETURN node2.name AS source, id(node2) AS source_id, type(r) AS relationship, id(r) AS relation_id, node1.name AS destination, id(node1) AS destination_id, similarity
          ORDER BY similarity DESC
          LIMIT toInteger($limit)
        `;

        const result = await session.run(cypherQuery, {
          n_embedding: nEmbedding,
          threshold: this.threshold,
          user_id: filters.user_id,
          limit: Math.floor(Number(limit)),
        });

        resultRelations.push(
          ...result.records.map((record) => ({
            source: record.get("source"),
            source_id: record.get("source_id").toString(),
            relationship: record.get("relationship"),
            relation_id: record.get("relation_id").toString(),
            destination: record.get("destination"),
            destination_id: record.get("destination_id").toString(),
            similarity: record.get("similarity"),
          })),
        );
      }
    } finally {
      await session.close();
    }
    return resultRelations;
  }

  private async getDeleteEntitiesFromSearchOutput(
    searchOutput: SearchResult[],
    data: string,
    filters: Filters,
  ): Promise<Entity[]> {
    const searchOutputString = formatEntities(searchOutput);
    const [systemPrompt, userPrompt] = getDeleteMessages(
      searchOutputString,
      data,
      filters.user_id,
    );

    const tools =
      this.llmProvider === "azure_openai_structured" || this.llmProvider === "openai_structured"
        ? [DELETE_MEMORY_STRUCT_TOOL_GRAPH]
        : [DELETE_MEMORY_TOOL_GRAPH];

    const memoryUpdates: LLMResponse = await this.llm.generateResponse({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      tools,
    });

    const toBeDeleted: Entity[] = [];
    for (const item of memoryUpdates.tool_calls) {
      if (item.name === "delete_graph_memory") {
        toBeDeleted.push(item.arguments);
      }
    }

    return this.removeSpacesFromEntities(toBeDeleted);
  }

  private async deleteEntities(
    toBeDeleted: Entity[],
    userId: string,
  ): Promise<DeleteResult[]> {
    const results: DeleteResult[] = [];
    const session = this.graph.session();
    try {
      for (const item of toBeDeleted) {
        const cypher = `
          MATCH (n:Entity {name: $source_name, user_id: $user_id})-[r:${item.relationship}]->(m:Entity {name: $dest_name, user_id: $user_id})
          DELETE r
          RETURN n.name AS source, m.name AS target, type(r) AS relationship
        `;
        const res = await session.run(cypher, {
          source_name: item.source,
          dest_name: item.destination,
          user_id: userId,
        });
        results.push(
          ...res.records.map((rec) => ({
            source: rec.get("source"),
            target: rec.get("target"),
            relationship: rec.get("relationship"),
          })),
        );
      }
    } finally {
      await session.close();
    }
    return results;
  }

  private async addEntities(
    toBeAdded: Entity[],
    userId: string,
    entityTypeMap: EntityTypeMap,
  ): Promise<any[]> {
    const results: any[] = [];
    const session = this.graph.session();
    try {
      for (const item of toBeAdded) {
        const sourceType = entityTypeMap[item.source] || "unknown";
        const destinationType = entityTypeMap[item.destination] || "unknown";
        const sourceEmbedding = await this.embeddingModel.embed(item.source);
        const destEmbedding = await this.embeddingModel.embed(item.destination);
        const sourceNodeSearchResult = await this.searchSourceNode(
          sourceEmbedding,
          userId,
          0.9,
        );
        const destinationNodeSearchResult = await this.searchDestinationNode(
          destEmbedding,
          userId,
          0.9,
        );

        let cypher: string;
        let params: any;

        if (!destinationNodeSearchResult.length && sourceNodeSearchResult.length) {
          cypher = `
            MATCH (source:Entity) WHERE id(source) = $source_id
            MERGE (destination:${destinationType}:Entity {name: $destination_name, user_id: $user_id})
            ON CREATE SET destination.created = timestamp(), destination.embedding = $destination_embedding
            MERGE (source)-[r:${item.relationship}]->(destination)
            ON CREATE SET r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            source_id: Number(sourceNodeSearchResult[0]["id(source_candidate)"]),
            destination_name: item.destination,
            destination_embedding: destEmbedding,
            user_id: userId,
          };
        } else if (destinationNodeSearchResult.length && !sourceNodeSearchResult.length) {
          cypher = `
            MATCH (destination:Entity) WHERE id(destination) = $destination_id
            MERGE (source:${sourceType}:Entity {name: $source_name, user_id: $user_id})
            ON CREATE SET source.created = timestamp(), source.embedding = $source_embedding
            MERGE (source)-[r:${item.relationship}]->(destination)
            ON CREATE SET r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            destination_id: Number(destinationNodeSearchResult[0]["id(destination_candidate)"]),
            source_name: item.source,
            source_embedding: sourceEmbedding,
            user_id: userId,
          };
        } else if (sourceNodeSearchResult.length && destinationNodeSearchResult.length) {
          cypher = `
            MATCH (source:Entity) WHERE id(source) = $source_id
            MATCH (destination:Entity) WHERE id(destination) = $destination_id
            MERGE (source)-[r:${item.relationship}]->(destination)
            ON CREATE SET r.created_at = timestamp(), r.updated_at = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            source_id: Number(sourceNodeSearchResult[0]["id(source_candidate)"]),
            destination_id: Number(destinationNodeSearchResult[0]["id(destination_candidate)"]),
            user_id: userId,
          };
        } else {
          cypher = `
            MERGE (n:${sourceType}:Entity {name: $source_name, user_id: $user_id})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:${destinationType}:Entity {name: $dest_name, user_id: $user_id})
            ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
            ON MATCH SET m.embedding = $dest_embedding
            MERGE (n)-[rel:${item.relationship}]->(m)
            ON CREATE SET rel.created = timestamp()
            RETURN n.name AS source, type(rel) AS relationship, m.name AS target
          `;
          params = {
            source_name: item.source,
            dest_name: item.destination,
            source_embedding: sourceEmbedding,
            dest_embedding: destEmbedding,
            user_id: userId,
          };
        }

        const res = await session.run(cypher, params);
        results.push(
          ...res.records.map((rec) => ({
            source: rec.get("source"),
            relationship: rec.get("relationship"),
            target: rec.get("target"),
          })),
        );
      }
    } finally {
      await session.close();
    }
    return results;
  }

  private removeSpacesFromEntities(entityList: Entity[]): Entity[] {
    return entityList.map((item) => ({
      source: item.source.toLowerCase().replace(/ /g, "_"),
      relationship: item.relationship.toLowerCase().replace(/ /g, "_"),
      destination: item.destination.toLowerCase().replace(/ /g, "_"),
    }));
  }

  private async searchSourceNode(
    sourceEmbedding: number[],
    userId: string,
    threshold: number = 0.9,
  ): Promise<any[]> {
    const session = this.graph.session();
    try {
      const cypher = `
        CALL vector_search.search("memzero", 1, $source_embedding) YIELD distance, node, similarity
        WITH node AS source_candidate, similarity
        WHERE source_candidate.user_id = $user_id AND similarity >= $threshold
        RETURN id(source_candidate) AS id
      `;
      const res = await session.run(cypher, {
        source_embedding: sourceEmbedding,
        user_id: userId,
        threshold,
      });
      return res.records.map((r) => ({ "id(source_candidate)": r.get("id").toInt() }));
    } finally {
      await session.close();
    }
  }

  private async searchDestinationNode(
    destinationEmbedding: number[],
    userId: string,
    threshold: number = 0.9,
  ): Promise<any[]> {
    const session = this.graph.session();
    try {
      const cypher = `
        CALL vector_search.search("memzero", 1, $destination_embedding) YIELD distance, node, similarity
        WITH node AS destination_candidate, similarity
        WHERE destination_candidate.user_id = $user_id AND similarity >= $threshold
        RETURN id(destination_candidate) AS id
      `;
      const res = await session.run(cypher, {
        destination_embedding: destinationEmbedding,
        user_id: userId,
        threshold,
      });
      return res.records.map((r) => ({ "id(destination_candidate)": r.get("id").toInt() }));
    } finally {
      await session.close();
    }
  }
}