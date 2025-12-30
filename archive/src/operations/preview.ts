import { SchemaValidator } from '../core/schema-validator.js';
import * as path from 'path';

/**
 * Validate an artifact without saving it
 * Allows testing before committing to the filesystem
 */
export async function validateArtifactOperation(
  content: any,
  type: 'spec' | 'conformance',
  namespace?: string,
  specDir: string = path.join(process.cwd(), 'spec')
): Promise<{ content: Array<{ type: string; text: string }> }> {
  const validator = new SchemaValidator(specDir);

  try {
    // Validate content structure
    if (!content || typeof content !== 'object') {
      return {
        content: [{
          type: 'text',
          text: '❌ Validation failed: Content must be a JSON object'
        }]
      };
    }

    // For conformance, check if it has proper structure
    if (type === 'conformance') {
      // Let base schema validation handle required fields
      const baseValidation = await validator.validateBaseSchema(content);
      if (!baseValidation.valid) {
        let response = '❌ Base validation failed:\n\n';
        for (const error of baseValidation.errors) {
          response += `  • ${error.message}`;
          if (error.path) {
            response += ` (at ${error.path})`;
          }
          response += '\n';
        }
        return {
          content: [{
            type: 'text',
            text: response
          }]
        };
      }

      // Determine conformance type from namespace
      const namespaceParts = content.namespace.split('.');
      const conformanceType = namespaceParts[namespaceParts.length - 1];

      // Validate against the appropriate schema
      const schemaName = `hologram.${conformanceType}.spec`;
      const validation = await validator.validateAgainstSchema(content, schemaName);

      if (!validation.valid) {
        let response = `❌ Validation failed for ${conformanceType} conformance:\n\n`;
        for (const error of validation.errors) {
          response += `  • ${error.message}`;
          if (error.path) {
            response += ` (at ${error.path})`;
          }
          response += '\n';
        }
        return {
          content: [{
            type: 'text',
            text: response
          }]
        };
      }
    }

    // For spec files, validate as JSON Schema
    if (type === 'spec') {
      if (!content.$schema) {
        return {
          content: [{
            type: 'text',
            text: '❌ Validation failed: Spec artifact missing $schema field'
          }]
        };
      }

      // Basic JSON Schema validation
      const baseValidation = await validator.validateBaseSchema(content);
      if (!baseValidation.valid) {
        let response = '❌ Validation failed for spec:\n\n';
        for (const error of baseValidation.errors) {
          response += `  • ${error.message}`;
          if (error.path) {
            response += ` (at ${error.path})`;
          }
          response += '\n';
        }
        return {
          content: [{
            type: 'text',
            text: response
          }]
        };
      }
    }

    // Success response with helpful info
    let response = '✅ Artifact validation successful!\n\n';
    response += `Type: ${type}\n`;

    if (type === 'conformance') {
      response += `Namespace: ${content.namespace}\n`;
      response += `Parent: ${content.parent}\n`;
      const conformanceType = content.namespace.split('.').pop();
      response += `Conformance Type: ${conformanceType}\n`;
    } else {
      response += `Schema ID: ${content.$id || 'not specified'}\n`;
    }

    response += '\nThis artifact is valid and can be submitted with submitArtifact()';

    return {
      content: [{
        type: 'text',
        text: response
      }]
    };

  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `❌ Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]
    };
  }
}

/**
 * Explain what validations are performed on a component
 */
export async function explainValidationOperation(
  namespace: string,
  specDir: string = path.join(process.cwd(), 'spec')
): Promise<{ content: Array<{ type: string; text: string }> }> {
  const validator = new SchemaValidator(specDir);

  try {
    let response = `VALIDATION EXPLANATION FOR: ${namespace}\n`;
    response += '=' .repeat(50) + '\n\n';

    // Get component model to understand requirements
    const componentModel = await validator.getComponentRequirements();

    if (!componentModel) {
      return {
        content: [{
          type: 'text',
          text: 'Component model not found. Cannot explain validation requirements.'
        }]
      };
    }

    response += '📋 VALIDATION CHECKS PERFORMED:\n\n';

    response += '1. INDEX FILE VALIDATION\n';
    response += '   • File exists: ' + namespace + '.index.json\n';
    response += '   • Has namespace field\n';
    response += '   • Has artifacts object\n';
    response += '   • All required conformance types present\n\n';

    response += '2. REQUIRED CONFORMANCE FILES\n';
    const conformanceReqs = componentModel.conformance_requirements || {};
    for (const [type, req] of Object.entries(conformanceReqs)) {
      const requirement = req as any;
      if (requirement.required) {
        response += `   • ${type}: ${requirement.description}\n`;
        response += `     - Schema: ${requirement.schema}\n`;
        response += `     - File pattern: ${namespace}.${type}.*.json\n`;
      }
    }
    response += '\n';

    response += '3. BASE SCHEMA VALIDATION (all files)\n';
    response += '   • Has namespace field matching expected pattern\n';
    response += '   • Has conformance boolean field\n';
    response += '   • Has version field (semantic versioning)\n';
    response += '   • Has description field\n\n';

    response += '4. SPEC VALIDATION\n';
    response += '   • Valid JSON Schema structure\n';
    response += '   • Has $schema field\n';
    response += '   • Defines properties and requirements\n\n';

    response += '5. CONFORMANCE VALIDATION\n';
    response += '   • Each conformance file validates against its schema\n';
    response += '   • Parent field references correct component\n';
    response += '   • Conformance flag is true\n';
    response += '   • Content matches expected structure\n\n';

    response += '6. CONTENT ADDRESSING\n';
    response += '   • Files use SHA256 hash in filename\n';
    response += '   • Content matches hash\n';
    response += '   • Index references correct artifacts\n\n';

    response += '💡 COMMON VALIDATION FAILURES:\n';
    response += '   • Missing required conformance files\n';
    response += '   • Incorrect namespace patterns\n';
    response += '   • Schema validation errors\n';
    response += '   • Missing required fields\n';
    response += '   • Type mismatches\n\n';

    response += 'Use validate({namespace: "' + namespace + '"}) to run actual validation';

    return {
      content: [{
        type: 'text',
        text: response
      }]
    };

  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `Error explaining validation: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]
    };
  }
}