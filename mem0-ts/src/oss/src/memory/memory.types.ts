import { Message } from "../types";
import { SearchFilters } from "../types";

export interface Entity {
  userId?: string;
  agentId?: string;
  runId?: string;
}

export interface AddMemoryOptions extends Entity {
  metadata?: Record<string, any>;
  filters?: SearchFilters;
  infer?: boolean;
}

export interface SearchMemoryOptions extends Entity {
  limit?: number;
  filters?: SearchFilters;
}

export interface GetAllMemoryOptions extends Entity {
  limit?: number;
}

export interface DeleteAllMemoryOptions extends Entity {}

export interface IMemoryGraph {
  add(
    data: string,
    filters: Record<string, any>
  ): Promise<{
    deleted_entities: any[];
    added_entities: any[];
    relations?: any[];
  }>;

  search(
    query: string,
    filters: Record<string, any>,
    limit?: number
  ): Promise<
    {
      source: string;
      relationship: string;
      destination: string;
      source_id?: string | number;
      destination_id?: string | number;
      relation_id?: string | number;
      similarity?: number;
    }[]
  >;

  deleteAll(filters: Record<string, any>): Promise<void>;

  getAll(
    filters: Record<string, any>,
    limit?: number
  ): Promise<
    {
      source: string;
      relationship: string;
      destination?: string;
    }[]
  >;
}
