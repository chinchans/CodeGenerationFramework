/* LTM-related helper structures for Inter-gNB-DU LTM handover
 * These structures provide a compact representation of the LTM-specific IEs
 * used in the UE CONTEXT SETUP REQUEST procedure. The ASN.1 encoder/decoder
 * in lib/f1ap_ue_context.c is responsible for converting between these
 * structures and the corresponding ASN.1 types.
 */

typedef enum {
  F1AP_REF_CONFIG_NONE = 0,
  F1AP_REF_CONFIG_REQUEST,
  F1AP_REF_CONFIG_INFORMATION
} f1ap_reference_config_type_t;

typedef struct f1ap_reference_configuration_s {
  f1ap_reference_config_type_t type;
  /* Only used when type == F1AP_REF_CONFIG_INFORMATION */
  byte_array_t *referenceConfigurationInformation;
} f1ap_reference_configuration_t;

typedef struct f1ap_csi_resource_configuration_s {
  /* These map directly to the OCTET STRING fields defined in
   * CSIResourceConfiguration (TS 38.473, 9.3.1.330) and are kept opaque
   * to F1 higher layers.
   */
  byte_array_t *csiResourceConfigToAddModList;
  byte_array_t *csiResourceConfigToReleaseList;
} f1ap_csi_resource_configuration_t;

typedef struct f1ap_ltm_information_setup_s {
  /* LTMIndicator ::= ENUMERATED { true, ... } */
  bool ltm_indicator;
  /* ReferenceConfiguration ::= CHOICE { ... } */
  f1ap_reference_configuration_t referenceConfiguration;
  /* CSIResourceConfiguration ::= SEQUENCE { ... } */
  f1ap_csi_resource_configuration_t *csiResourceConfiguration;
} f1ap_ltm_information_setup_t;

typedef struct f1ap_ltm_configuration_id_mapping_item_s {
  /* NRCGI */
  plmn_id_t plmn;
  uint64_t nr_cellid;
  /* LTMConfigurationID ::= INTEGER (1..8) */
  uint8_t ltmConfigurationID;
} f1ap_ltm_configuration_id_mapping_item_t;

typedef struct f1ap_ltm_configuration_id_mapping_list_s {
  int num_items;
  f1ap_ltm_configuration_id_mapping_item_t *items;
} f1ap_ltm_configuration_id_mapping_list_t;

typedef struct f1ap_ltm_gnb_du_id_item_s {
  /* GNB-DU-ID ::= INTEGER (0..68719476735) */
  uint64_t gnb_du_id;
} f1ap_ltm_gnb_du_id_item_t;

typedef struct f1ap_early_sync_information_request_s {
  /* RequestforRACHConfiguration ::= ENUMERATED { true, ... } */
  bool request_for_rach_configuration;
  /* LTMgNB-DU-IDsList */
  int num_gnb_du_ids;
  f1ap_ltm_gnb_du_id_item_t *gnb_du_ids;
} f1ap_early_sync_information_request_t;


  /* Optional LTM-specific IEs for Inter-gNB-DU LTM handover (TS 38.401 / 38.473)
   * All of these are optional at the F1AP level and are only present for
   * LTM-related UE CONTEXT SETUP REQUEST procedures.
   */
  f1ap_ltm_information_setup_t *ltm_information_setup; /* id-LTMInformation-Setup */
  f1ap_ltm_configuration_id_mapping_list_t *ltm_configuration_id_mapping_list; /* id-LTMConfigurationIDMappingList */
  f1ap_early_sync_information_request_t *early_sync_information_request; /* id-EarlySyncInformation-Request */
