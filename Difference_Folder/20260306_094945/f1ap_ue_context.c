  /* optional: LTMInformation-Setup (Inter-gNB-DU LTM handover support) */
  if (req->ltm_information_setup) {
    const f1ap_ltm_information_setup_t *ltm = req->ltm_information_setup;
    asn1cSequenceAdd(out->protocolIEs.list, F1AP_UEContextSetupRequestIEs_t, ie);
    ie->id = F1AP_ProtocolIE_ID_id_LTMInformation_Setup;
    ie->criticality = F1AP_Criticality_reject;
    ie->value.present = F1AP_UEContextSetupRequestIEs__value_PR_LTMInformation_Setup;
    F1AP_LTMInformation_Setup_t *asn_ltm = &ie->value.choice.LTMInformation_Setup;

    /* LTMIndicator */
    asn_ltm->lTMIndicator = ltm->ltm_indicator ? F1AP_LTMIndicator_true : F1AP_LTMIndicator_true;

    /* ReferenceConfiguration */
    F1AP_ReferenceConfiguration_t *rc = &asn_ltm->referenceConfiguration;
    switch (ltm->referenceConfiguration.type) {
      case F1AP_REF_CONFIG_REQUEST:
        rc->present = F1AP_ReferenceConfiguration_PR_requestforReferenceConfiguration;
        break;
      case F1AP_REF_CONFIG_INFORMATION:
        rc->present = F1AP_ReferenceConfiguration_PR_referenceConfigurationInformation;
        if (ltm->referenceConfiguration.referenceConfigurationInformation) {
          byte_array_t *ba = ltm->referenceConfiguration.referenceConfigurationInformation;
          OCTET_STRING_fromBuf(&rc->choice.referenceConfigurationInformation,
                               (const char *)ba->buf, ba->len);
        }
        break;
      case F1AP_REF_CONFIG_NONE:
      default:
        rc->present = F1AP_ReferenceConfiguration_PR_NOTHING;
        break;
    }

    /* CSIResourceConfiguration (only if provided) */
    if (ltm->csiResourceConfiguration) {
      asn1cCalloc(asn_ltm->cSIResourceConfiguration, asn_csi);
      const f1ap_csi_resource_configuration_t *csi = ltm->csiResourceConfiguration;
      if (csi->csiResourceConfigToAddModList) {
        byte_array_t *ba = csi->csiResourceConfigToAddModList;
        OCTET_STRING_fromBuf(&asn_csi->cSIResourceConfigToAddModList,
                             (const char *)ba->buf, ba->len);
      }
      if (csi->csiResourceConfigToReleaseList) {
        byte_array_t *ba = csi->csiResourceConfigToReleaseList;
        OCTET_STRING_fromBuf(&asn_csi->cSIResourceConfigToReleaseList,
                             (const char *)ba->buf, ba->len);
      }
    }
  }

  /* optional: LTMConfigurationIDMappingList */
  if (req->ltm_configuration_id_mapping_list && req->ltm_configuration_id_mapping_list->num_items > 0) {
    const f1ap_ltm_configuration_id_mapping_list_t *lst = req->ltm_configuration_id_mapping_list;
    asn1cSequenceAdd(out->protocolIEs.list, F1AP_UEContextSetupRequestIEs_t, ie);
    ie->id = F1AP_ProtocolIE_ID_id_LTMConfigurationIDMappingList;
    ie->criticality = F1AP_Criticality_reject;
    ie->value.present = F1AP_UEContextSetupRequestIEs__value_PR_LTMConfigurationIDMappingList;
    F1AP_LTMConfigurationIDMappingList_t *asn_list = &ie->value.choice.LTMConfigurationIDMappingList;

    for (int i = 0; i < lst->num_items; ++i) {
      const f1ap_ltm_configuration_id_mapping_item_t *item = &lst->items[i];
      asn1cSequenceAdd(asn_list->list, F1AP_LTMConfigurationIDMapping_Item_t, asn_item);

      /* lTMCellID (NRCGI) */
      MCC_MNC_TO_PLMNID(item->plmn.mcc, item->plmn.mnc, item->plmn.mnc_digit_length,
                        &asn_item->lTMCellID.pLMN_Identity);
      NR_CELL_ID_TO_BIT_STRING(item->nr_cellid, &asn_item->lTMCellID.nRCellIdentity);

      /* lTMConfigurationID (INTEGER 1..8) */
      asn_item->lTMConfigurationID = item->ltmConfigurationID;
    }
  }

  /* optional: EarlySyncInformation-Request */
  if (req->early_sync_information_request) {
    const f1ap_early_sync_information_request_t *es = req->early_sync_information_request;
    asn1cSequenceAdd(out->protocolIEs.list, F1AP_UEContextSetupRequestIEs_t, ie);
    ie->id = F1AP_ProtocolIE_ID_id_EarlySyncInformation_Request;
    ie->criticality = F1AP_Criticality_ignore;
    ie->value.present = F1AP_UEContextSetupRequestIEs__value_PR_EarlySyncInformation_Request;
    F1AP_EarlySyncInformation_Request_t *asn_es = &ie->value.choice.EarlySyncInformation_Request;

    /* requestforRACHConfiguration */
    asn_es->requestforRACHConfiguration = es->request_for_rach_configuration ? F1AP_RequestforRACHConfiguration_true : F1AP_RequestforRACHConfiguration_true;

    /* lTMgNB-DU-IDsList */
    for (int i = 0; i < es->num_gnb_du_ids; ++i) {
      const f1ap_ltm_gnb_du_id_item_t *id_item = &es->gnb_du_ids[i];
      asn1cSequenceAdd(asn_es->lTMgNB_DU_IDsList.list, F1AP_LTMgNB_DU_IDs_Item_t, asn_item);
      asn_item->lTMgNB_DU_ID = id_item->gnb_du_id;
    }
  }

      case F1AP_ProtocolIE_ID_id_LTMInformation_Setup: {
        _F1_EQ_CHECK_INT(ie->value.present, F1AP_UEContextSetupRequestIEs__value_PR_LTMInformation_Setup);
        const F1AP_LTMInformation_Setup_t *asn_ltm = &ie->value.choice.LTMInformation_Setup;
        out->ltm_information_setup = calloc_or_fail(1, sizeof(*out->ltm_information_setup));
        f1ap_ltm_information_setup_t *ltm = out->ltm_information_setup;

        /* LTMIndicator (only 'true' is currently defined) */
        ltm->ltm_indicator = (asn_ltm->lTMIndicator == F1AP_LTMIndicator_true);

        /* ReferenceConfiguration */
        F1AP_ReferenceConfiguration_t *rc = (F1AP_ReferenceConfiguration_t *)&asn_ltm->referenceConfiguration;
        switch (rc->present) {
          case F1AP_ReferenceConfiguration_PR_requestforReferenceConfiguration:
            ltm->referenceConfiguration.type = F1AP_REF_CONFIG_REQUEST;
            break;
          case F1AP_ReferenceConfiguration_PR_referenceConfigurationInformation: {
            ltm->referenceConfiguration.type = F1AP_REF_CONFIG_INFORMATION;
            OCTET_STRING_t *os = &rc->choice.referenceConfigurationInformation;
            ltm->referenceConfiguration.referenceConfigurationInformation =
                malloc_or_fail(sizeof(*ltm->referenceConfiguration.referenceConfigurationInformation));
            *ltm->referenceConfiguration.referenceConfigurationInformation =
                create_byte_array(os->size, os->buf);
            } break;
          case F1AP_ReferenceConfiguration_PR_NOTHING:
          default:
            ltm->referenceConfiguration.type = F1AP_REF_CONFIG_NONE;
            break;
        }

        /* CSIResourceConfiguration */
        if (asn_ltm->cSIResourceConfiguration) {
          const F1AP_CSIResourceConfiguration_t *asn_csi = asn_ltm->cSIResourceConfiguration;
          ltm->csiResourceConfiguration = calloc_or_fail(1, sizeof(*ltm->csiResourceConfiguration));
          f1ap_csi_resource_configuration_t *csi = ltm->csiResourceConfiguration;

          if (asn_csi->cSIResourceConfigToAddModList.buf && asn_csi->cSIResourceConfigToAddModList.size > 0) {
            csi->csiResourceConfigToAddModList = malloc_or_fail(sizeof(*csi->csiResourceConfigToAddModList));
            *csi->csiResourceConfigToAddModList =
                create_byte_array(asn_csi->cSIResourceConfigToAddModList.size,
                                  asn_csi->cSIResourceConfigToAddModList.buf);
          }
          if (asn_csi->cSIResourceConfigToReleaseList.buf && asn_csi->cSIResourceConfigToReleaseList.size > 0) {
            csi->csiResourceConfigToReleaseList = malloc_or_fail(sizeof(*csi->csiResourceConfigToReleaseList));
            *csi->csiResourceConfigToReleaseList =
                create_byte_array(asn_csi->cSIResourceConfigToReleaseList.size,
                                  asn_csi->cSIResourceConfigToReleaseList.buf);
          }
        }
        } break;
      case F1AP_ProtocolIE_ID_id_LTMConfigurationIDMappingList: {
        _F1_EQ_CHECK_INT(ie->value.present, F1AP_UEContextSetupRequestIEs__value_PR_LTMConfigurationIDMappingList);
        const F1AP_LTMConfigurationIDMappingList_t *asn_list = &ie->value.choice.LTMConfigurationIDMappingList;
        out->ltm_configuration_id_mapping_list =
            calloc_or_fail(1, sizeof(*out->ltm_configuration_id_mapping_list));
        f1ap_ltm_configuration_id_mapping_list_t *lst = out->ltm_configuration_id_mapping_list;
        lst->num_items = asn_list->list.count;
        if (lst->num_items > 0) {
          lst->items = calloc_or_fail(lst->num_items, sizeof(*lst->items));
          for (int j = 0; j < lst->num_items; ++j) {
            F1AP_LTMConfigurationIDMapping_Item_t *asn_item =
                (F1AP_LTMConfigurationIDMapping_Item_t *)asn_list->list.array[j];
            f1ap_ltm_configuration_id_mapping_item_t *item = &lst->items[j];
            const F1AP_NRCGI_t *nrcgi = &asn_item->lTMCellID;
            PLMNID_TO_MCC_MNC(&nrcgi->pLMN_Identity,
                              item->plmn.mcc,
                              item->plmn.mnc,
                              item->plmn.mnc_digit_length);
            BIT_STRING_TO_NR_CELL_IDENTITY(&nrcgi->nRCellIdentity, item->nr_cellid);
            item->ltmConfigurationID = asn_item->lTMConfigurationID;
          }
        }
        } break;
      case F1AP_ProtocolIE_ID_id_EarlySyncInformation_Request: {
        _F1_EQ_CHECK_INT(ie->value.present, F1AP_UEContextSetupRequestIEs__value_PR_EarlySyncInformation_Request);
        const F1AP_EarlySyncInformation_Request_t *asn_es = &ie->value.choice.EarlySyncInformation_Request;
        out->early_sync_information_request =
            calloc_or_fail(1, sizeof(*out->early_sync_information_request));
        f1ap_early_sync_information_request_t *es = out->early_sync_information_request;

        es->request_for_rach_configuration =
            (asn_es->requestforRACHConfiguration == F1AP_RequestforRACHConfiguration_true);

        int count = asn_es->lTMgNB_DU_IDsList.list.count;
        es->num_gnb_du_ids = count;
        if (count > 0) {
          es->gnb_du_ids = calloc_or_fail(count, sizeof(*es->gnb_du_ids));
          for (int j = 0; j < count; ++j) {
            F1AP_LTMgNB_DU_IDs_Item_t *asn_item =
                (F1AP_LTMgNB_DU_IDs_Item_t *)asn_es->lTMgNB_DU_IDsList.list.array[j];
            es->gnb_du_ids[j].gnb_du_id = asn_item->lTMgNB_DU_ID;
          }
        }
        } break;

  /* free LTMInformation-Setup */
  if (req->ltm_information_setup) {
    f1ap_ltm_information_setup_t *ltm = req->ltm_information_setup;
    if (ltm->referenceConfiguration.referenceConfigurationInformation) {
      free_byte_array(ltm->referenceConfiguration.referenceConfigurationInformation);
      free(ltm->referenceConfiguration.referenceConfigurationInformation);
    }
    if (ltm->csiResourceConfiguration) {
      f1ap_csi_resource_configuration_t *csi = ltm->csiResourceConfiguration;
      if (csi->csiResourceConfigToAddModList) {
        free_byte_array(csi->csiResourceConfigToAddModList);
        free(csi->csiResourceConfigToAddModList);
      }
      if (csi->csiResourceConfigToReleaseList) {
        free_byte_array(csi->csiResourceConfigToReleaseList);
        free(csi->csiResourceConfigToReleaseList);
      }
      free(csi);
    }
    free(ltm);
    req->ltm_information_setup = NULL;
  }

  /* free LTMConfigurationIDMappingList */
  if (req->ltm_configuration_id_mapping_list) {
    free(req->ltm_configuration_id_mapping_list->items);
    free(req->ltm_configuration_id_mapping_list);
    req->ltm_configuration_id_mapping_list = NULL;
  }

  /* free EarlySyncInformation-Request */
  if (req->early_sync_information_request) {
    free(req->early_sync_information_request->gnb_du_ids);
    free(req->early_sync_information_request);
    req->early_sync_information_request = NULL;
  }
