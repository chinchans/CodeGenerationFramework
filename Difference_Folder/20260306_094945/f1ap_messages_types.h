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

/* ----------------------------- */
/* LTM-related UEContextSetupResponse IEs (R18.6.0) */
/* ----------------------------- */

typedef struct f1ap_preamble_index_list_item_s {
  /* PreambleIndex ::= INTEGER (0..63) */
  uint8_t preamble_index;
} f1ap_preamble_index_list_item_t;

typedef struct f1ap_preamble_index_list_s {
  int num_items;
  f1ap_preamble_index_list_item_t *items;
} f1ap_preamble_index_list_t;

typedef struct f1ap_ltm_gnb_du_id_preamble_index_item_s {
  /* lTMgNB-DU-ID ::= GNB-DU-ID */
  uint64_t gnb_du_id;
  /* optional preambleIndexList */
  f1ap_preamble_index_list_t *preamble_index_list;
} f1ap_ltm_gnb_du_id_preamble_index_item_t;

typedef struct f1ap_ltm_gnb_du_ids_preamble_index_list_s {
  int num_items;
  f1ap_ltm_gnb_du_id_preamble_index_item_t *items;
} f1ap_ltm_gnb_du_ids_preamble_index_list_t;

typedef struct f1ap_early_ul_sync_config_s {
  /* RACHConfiguration ::= OCTET STRING */
  byte_array_t rach_configuration;
  /* optional: LTMgNB-DU-IDs-PreambleIndexList */
  f1ap_ltm_gnb_du_ids_preamble_index_list_t *ltm_gnb_du_ids_preamble_index_list;
} f1ap_early_ul_sync_config_t;

typedef struct f1ap_early_sync_information_s {
  /* TCIStatesConfigurationsList ::= OCTET STRING */
  byte_array_t tci_states_configurations_list;
  /* optional */
  f1ap_early_ul_sync_config_t *early_ul_sync_config;
  /* optional */
  f1ap_early_ul_sync_config_t *early_ul_sync_config_sul;
} f1ap_early_sync_information_t;

typedef enum {
  F1AP_SSB_POSITIONS_IN_BURST_NONE = 0,
  F1AP_SSB_POSITIONS_IN_BURST_SHORT_BITMAP,
  F1AP_SSB_POSITIONS_IN_BURST_MEDIUM_BITMAP,
  F1AP_SSB_POSITIONS_IN_BURST_LONG_BITMAP
} f1ap_ssb_positions_in_burst_type_t;

typedef struct f1ap_ssb_positions_in_burst_s {
  f1ap_ssb_positions_in_burst_type_t type;
  /* For short/medium: bitmap is stored in the LSBs. For long: full 64-bit bitmap. */
  uint64_t bitmap;
} f1ap_ssb_positions_in_burst_t;

typedef struct f1ap_ssb_tf_configuration_s {
  uint32_t ssb_frequency; /* 0..3279165 */
  /* Subcarrier spacing in kHz (15/30/60/120/240/480/960). */
  uint16_t ssb_subcarrier_spacing_khz;
  int8_t ssb_transmit_power; /* -60..50 */
  /* Periodicity in ms (5/10/20/40/80/160). */
  uint16_t ssb_periodicity_ms;
  uint8_t ssb_half_frame_offset; /* 0..1 */
  uint8_t ssb_sfn_offset;        /* 0..15 */
  /* optional */
  f1ap_ssb_positions_in_burst_t *ssb_position_in_burst;
} f1ap_ssb_tf_configuration_t;

typedef struct f1ap_ssb_information_item_s {
  f1ap_ssb_tf_configuration_t ssb_configuration;
  uint16_t pci_nr; /* NRPCI 0..1007 */
} f1ap_ssb_information_item_t;

typedef struct f1ap_ssb_information_s {
  int num_items;
  f1ap_ssb_information_item_t *items;
} f1ap_ssb_information_t;

typedef struct f1ap_ltm_configuration_s {
  /* mandatory */
  f1ap_ssb_information_t ssb_information;

  /* optional: ReferenceConfigurationInformation ::= OCTET STRING */
  byte_array_t *reference_configuration_information;

  /* optional: CompleteCandidateConfigurationIndicator ::= ENUMERATED { complete, ... } */
  bool *complete_candidate_configuration_indicator;

  /* optional: LTMCFRAResourceConfig ::= OCTET STRING */
  byte_array_t *ltm_cfra_resource_config;
  byte_array_t *ltm_cfra_resource_config_sul;
} f1ap_ltm_configuration_t;


  /* Optional LTM-specific IEs for Inter-gNB-DU LTM handover (TS 38.401 / 38.473)
   * All of these are optional at the F1AP level and are only present for
   * LTM-related UE CONTEXT SETUP REQUEST procedures.
   */
  f1ap_ltm_information_setup_t *ltm_information_setup; /* id-LTMInformation-Setup */
  f1ap_ltm_configuration_id_mapping_list_t *ltm_configuration_id_mapping_list; /* id-LTMConfigurationIDMappingList */
  f1ap_early_sync_information_request_t *early_sync_information_request; /* id-EarlySyncInformation-Request */

  /* Optional LTM-related IEs for Inter-gNB-DU LTM handover (R18.6.0) */
  f1ap_early_sync_information_t *early_sync_information; /* id-EarlySyncInformation */
  f1ap_ltm_configuration_t *ltm_configuration;           /* id-LTMConfiguration */
